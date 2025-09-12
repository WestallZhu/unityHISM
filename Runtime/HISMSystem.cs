
using Renderloom.Rendering;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using Renderloom.HISM.Streaming;
using FrustumPlanes = Renderloom.Rendering.FrustumPlanes;

namespace Renderloom.HISM
{
    #region Data Structures (original or lightly extended)
    public struct HISMPrimitive
    {
        public AABB BoundingBox;
        public MeshMaterialIndexCount meshMaterialLods; // index + count (LOD entries)
        // public int submeshIndex;
    }

    public struct HISMNode
    {
        public AABB BoundingBox;
        public int LeftChildIndex;
        public int RightChildIndex;
        public int FirstInstance;
        public int LastInstance;
    }

    public struct MeshMaterialNode
    {
        public BatchMaterialID batchMaterialID;
        public BatchMeshID batchMeshID;
        public int nextIndex; // for freelist
    }

    public struct MeshMaterialIndexCount
    {
        public int index; // start index of MeshMaterialNode linear array
        public int count; // LOD count
    }

    public struct ChunkProperty
    {
        public int TypeIndex;
        public int ValueSizeBytesCPU;
        public int ValueSizeBytesGPU;
        public int GPUDataBegin;
    }

    public struct MaterialPropertyType
    {
        public int TypeIndex;
        public int NameID;
        public short SizeBytesCPU;
        public short SizeBytesGPU;
    }

    public static class TypeId<T>
    {
        public static readonly int Value = typeof(T) switch
        {
            var t when t == typeof(int) => 0,
            var t when t == typeof(float) => 1,
            var t when t == typeof(float2) => 2,
            var t when t == typeof(float4) => 3,
            _ => -1
        };
    }

    public unsafe struct HIMRRenderArray
    {
        static uint RenderHash(List<Material> matList, List<Mesh> meshList)
        {
            uint hash = 17;
            for (int i = 0; i < matList.Count; i++)
            {
                hash = hash * 31 + (uint)matList[i].GetInstanceID();
                hash = hash * 31 + (uint)meshList[i].GetInstanceID();
            }
            return hash;
        }

        const int MaxLod = 4;
        public UnsafeList<MeshMaterialNode> _meshMaterialList;
        fixed int _freeNode[MaxLod];

        int _freeNodeCount;
        int _allocatedCount;

        public NativeHashMap<uint, MeshMaterialIndexCount> _rendererHashMap;

        private BatchRendererGroup _batchRendererGroup;

        public void Init(BatchRendererGroup brg)
        {
            _freeNodeCount = 0;
            _allocatedCount = 0;
            for (int i = 0; i < MaxLod; i++)
                _freeNode[i] = -1;

            _batchRendererGroup = brg;
        }

        public MeshMaterialIndexCount GetOrCreateRender(List<Material> matList, List<Mesh> meshList)
        {
            uint hash = RenderHash(matList, meshList);

            if (!_rendererHashMap.TryGetValue(hash, out var renderer))
            {
                renderer = AllocateRender(matList.Count);
                _rendererHashMap[hash] = renderer;

                MeshMaterialNode* p = _meshMaterialList.Ptr;
                for (int i = renderer.index; i < renderer.index + renderer.count; i++)
                {
                    p[i].batchMeshID = _batchRendererGroup.RegisterMesh(meshList[i]);
                    p[i].batchMaterialID = _batchRendererGroup.RegisterMaterial(matList[i]);
                }
            }

            return renderer;
        }

        public bool GetRender(uint hash, ref MeshMaterialIndexCount renderer)
        {
            bool ret = _rendererHashMap.TryGetValue(hash, out renderer);
            if (!ret)
            {
                renderer.count = 0;
                renderer.index = -1;
            }
            return ret;
        }

        MeshMaterialIndexCount AllocateRender(int count)
        {
            int index = -1;
            if (_freeNode[count] >= 0)
            {
                index = _freeNode[count];
                _freeNode[count] = _meshMaterialList[index].nextIndex;
                _freeNodeCount -= count;
            }
            else
            {
                index = _meshMaterialList.Length;
                _meshMaterialList.Resize(index + count);
            }
            _allocatedCount += count;

            return new MeshMaterialIndexCount { index = index, count = count };
        }

        void FreeRender(ref MeshMaterialIndexCount renderer)
        {
            int index = renderer.index;
            int count = renderer.count;

            if (index < 0 || count <= 0)
                return;

            for (int i = renderer.index; i < renderer.index + renderer.count; i++)
            {
                _batchRendererGroup.UnregisterMesh(_meshMaterialList[i].batchMeshID);
                _batchRendererGroup.UnregisterMaterial(_meshMaterialList[i].batchMaterialID);
            }

            _allocatedCount -= count;
            _freeNodeCount += count;

            _meshMaterialList.Ptr[index].nextIndex = _freeNode[count];
            _freeNode[count] = index;

            renderer.count = 0;
            renderer.index = -1;
        }
    }

    internal struct VisibleCmd
    {
        public int lod;
        public int rendererIndex; // Reserved: set during Emit
        public int first;
        public int count;
        public int chunkIndex;
    }
    #endregion

    #region Culling + LOD Jobs (revised)
    [BurstCompile]
    public unsafe struct HISMCullAndLodJob : IJobParallelFor
    {
        [ReadOnly] public LODGroupExtensions.LODParams LODParams;
        [ReadOnly] public NativeArray<FrustumPlanes.PlanePacket4> Packet4;
        [ReadOnly] public NativeArray<HISMChunk> Chunks;

        public NativeStream.Writer Writer;
        public float4 LodThresholdSq; 

        public void Execute(int i)
        {
            var chunk = Chunks[i];
            var state = FrustumPlanes.Intersect2(Packet4, chunk.BoundingBox);
            if (state == FrustumPlanes.IntersectResult.Out) return;

            bool chunkSingleLod = TryWholeBoxSingleLod_Sq(chunk.BoundingBox, LODParams.cameraPos, LodThresholdSq, out int chunkLod);

            int primitiveLength = chunk.Primitives.Length;
            var primitives = (HISMPrimitive*)chunk.Primitives.GetUnsafePtr();

            if (state == FrustumPlanes.IntersectResult.In && chunkSingleLod)
            {
                Writer.BeginForEachIndex(i);
                Writer.Write(new VisibleCmd
                {
                    lod = chunkLod,
                    chunkIndex = i,
                    first = 0,
                    count = primitiveLength
                });
                Writer.EndForEachIndex();
                return;
            }

            if (state == FrustumPlanes.IntersectResult.Partial && chunkSingleLod)
            {
                Writer.BeginForEachIndex(i);
                int start = -1;
                int count = 0;
                for (int j = 0; j < primitiveLength; ++j)
                {
                    var prim = primitives[j];
                    if (FrustumPlanes.Intersect2(Packet4, prim.BoundingBox) == FrustumPlanes.IntersectResult.Out)
                    {
                        if (count > 0)
                        {
                            Writer.Write(new VisibleCmd { lod = chunkLod, chunkIndex = i, first = start, count = count });
                        }
                        start = -1;
                        count = 0;
                        continue;
                    }
                    else
                    {
                        if (start == -1) start = j;
                        count++;
                    }
                }

                if (count > 0)
                {
                    Writer.Write(new VisibleCmd { lod = chunkLod, chunkIndex = i, first = start, count = count });
                }

                Writer.EndForEachIndex();
                return;
            }

            var nodes = (HISMNode*)chunk.BVHTree.GetUnsafePtr();
            int* stack = stackalloc int[128]; int sp = 0; stack[sp++] = 0; // root

            Writer.BeginForEachIndex(i);
            while (sp > 0)
            {
                int nodeIdx = stack[--sp];
                var node = nodes[nodeIdx];

                bool nodeSingleLod = TryWholeBoxSingleLod_Sq(node.BoundingBox, LODParams.cameraPos, LodThresholdSq, out int nodeLod);

                var nstate = FrustumPlanes.Intersect2(Packet4, node.BoundingBox);
                if (nstate == FrustumPlanes.IntersectResult.Out) continue;

                if (nstate == FrustumPlanes.IntersectResult.In && nodeSingleLod)
                {
                    int first = node.FirstInstance;
                    int count = node.LastInstance - node.FirstInstance;
                    if (count > 0)
                    {
                        Writer.Write(new VisibleCmd { lod = nodeLod, chunkIndex = i, first = first, count = count });
                    }
                    continue;
                }

                // Leaf: partial-intersection + single LOD fast range output (fix)
                if (nstate == FrustumPlanes.IntersectResult.Partial && nodeSingleLod && node.LeftChildIndex < 0)
                {
                    int start = -1;
                    int count = 0;

                    for (int j = node.FirstInstance; j < node.LastInstance; ++j)
                    {
                        var prim = primitives[j];
                        if (FrustumPlanes.Intersect2(Packet4, prim.BoundingBox) == FrustumPlanes.IntersectResult.Out)
                        {
                            if (count > 0)
                            {
                                Writer.Write(new VisibleCmd { lod = nodeLod, chunkIndex = i, first = start, count = count });
                                count = 0; start = -1;
                            }
                            continue;
                        }
                        else
                        {
                            if (start == -1) start = j;
                            count++;
                        }
                    }

                    if (count > 0)
                    {
                        Writer.Write(new VisibleCmd { lod = nodeLod, chunkIndex = i, first = start, count = count });
                    }

                    continue;
                }

                if (node.LeftChildIndex >= 0)
                {
                    stack[sp++] = node.LeftChildIndex;
                    stack[sp++] = node.RightChildIndex;
                    continue;
                }

                for (int j = node.FirstInstance; j < node.LastInstance; ++j)
                {
                    var prim = primitives[j];
                    if (FrustumPlanes.Intersect2(Packet4, prim.BoundingBox) == FrustumPlanes.IntersectResult.Out)
                        continue;

                    float distSq = math.distancesq(LODParams.cameraPos, prim.BoundingBox.Center);
                    int lod = SelectLodByDistanceSq(distSq, LodThresholdSq);
                    Writer.Write(new VisibleCmd { lod = lod, chunkIndex = i, first = j, count = 1 });
                }
            }
            Writer.EndForEachIndex();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int SelectLodByDistanceSq(float distSq, float4 thresholdsSq)
        {
            if (distSq < thresholdsSq.x) return 0;
            if (distSq < thresholdsSq.y) return 1;
            if (distSq < thresholdsSq.z) return 2;
            return 3;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void DistanceSqRangeToAABB(float3 cam, in AABB aabb, out float dminSq, out float dmaxSq)
        {
            float3 min = aabb.Min, max = aabb.Max;
            float3 p = math.clamp(cam, min, max);
            dminSq = math.lengthsq(cam - p);

            float3 q;
            q.x = (math.abs(cam.x - min.x) > math.abs(cam.x - max.x)) ? min.x : max.x;
            q.y = (math.abs(cam.y - min.y) > math.abs(cam.y - max.y)) ? min.y : max.y;
            q.z = (math.abs(cam.z - min.z) > math.abs(cam.z - max.z)) ? min.z : max.z;
            dmaxSq = math.lengthsq(cam - q);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool TryWholeBoxSingleLod_Sq(in AABB box, float3 cam, in float4 thresholdsSq, out int lod)
        {
            DistanceSqRangeToAABB(cam, box, out float dminSq, out float dmaxSq);
            const float rel = 0.02f; // slight relaxation to avoid jitter
            dminSq *= (1f - rel) * (1f - rel);
            dmaxSq *= (1f + rel) * (1f + rel);

            int lmin = SelectLodByDistanceSq(dminSq, thresholdsSq);
            int lmax = SelectLodByDistanceSq(dmaxSq, thresholdsSq);
            if (lmin == lmax) { lod = lmin; return true; }
            lod = -1; return false;
        }
    }
    #endregion

    #region Runtime mapping: per-chunk instance offset within batch
    public struct ChunkRuntime
    {
        public int BatchIndex;         // == BatchID.value
        public int ChunkOffsetInBatch; // instance start within this batch
    }
    #endregion

    #region Parallel Emit + cross-range command merge Job
    [BurstCompile]
    public unsafe struct HISMCountRunsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<HISMChunk> Chunks;
        [ReadOnly] public NativeArray<ChunkRuntime> ChunkRuntimes;
        [ReadOnly] public NativeStream.Reader Reader;

        public NativeArray<int> RunCounts;     // per-chunk command count (merged by renderer)
        public NativeArray<int> VisibleCounts; // per-chunk visible instance count

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int ClampLod(in MeshMaterialIndexCount mm, int lod)
            => math.clamp(lod, 0, math.max(0, mm.count - 1));

        public void Execute(int ci)
        {
            var reader = Reader;
            int n = reader.BeginForEachIndex(ci);

            var chunk = Chunks[ci];
            var prims = (HISMPrimitive*)chunk.Primitives.GetUnsafePtr();
            int batchIndex = ChunkRuntimes[ci].BatchIndex;

            int runCount = 0;
            int visCount = 0;

            bool havePrev = false;
            int prevMM = -1;
            int prevBatch = -1;

            for (int k = 0; k < n; ++k)
            {
                var v = reader.Read<VisibleCmd>();
                int end = v.first + v.count;

                for (int j = v.first; j < end; ++j)
                {
                    visCount++;

                    var mm = prims[j].meshMaterialLods;
                    int mmIdx = mm.index + ClampLod(mm, v.lod);

                    if (!havePrev || mmIdx != prevMM || batchIndex != prevBatch)
                    {
                        runCount++;
                        havePrev = true;
                        prevMM = mmIdx;
                        prevBatch = batchIndex;
                    }
                }
            }

            reader.EndForEachIndex();

            RunCounts[ci] = runCount;
            VisibleCounts[ci] = visCount;
        }
    }

    [BurstCompile]
    public struct HISMExclusiveScanJob : IJob
    {
        [ReadOnly] public NativeArray<int> RunCounts;
        [ReadOnly] public NativeArray<int> VisibleCounts;

        public NativeArray<int> CmdOffsets;  // exclusive prefix
        public NativeArray<int> VisOffsets;  // exclusive prefix

        public void Execute()
        {
            int n = RunCounts.Length;

            int accCmd = 0;
            int accVis = 0;
            for (int i = 0; i < n; ++i)
            {
                CmdOffsets[i] = accCmd;
                VisOffsets[i] = accVis;
                accCmd += RunCounts[i];
                accVis += VisibleCounts[i];
            }
        }
    }

    [BurstCompile]
    public unsafe struct HISMAllocateOutputJob : IJob
    {
        [ReadOnly] public NativeArray<int> RunCounts;
        [ReadOnly] public NativeArray<int> VisibleCounts;

        [NativeDisableUnsafePtrRestriction] public BatchCullingOutputDrawCommands* Out;

        public void Execute()
        {
            int n = RunCounts.Length;
            int totalCmd = 0;
            int totalVis = 0;
            for (int i = 0; i < n; ++i) { totalCmd += RunCounts[i]; totalVis += VisibleCounts[i]; }

            // draw ranges: at least 1
            int rangeAlign = UnsafeUtility.AlignOf<BatchDrawRange>();
            Out->drawRangeCount = 1;
            Out->drawRanges = (BatchDrawRange*)UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<BatchDrawRange>() * Out->drawRangeCount, rangeAlign, Allocator.TempJob);

            // visibleInstances
            Out->visibleInstanceCount = totalVis;
            if (totalVis > 0)
            {
                int viAlign = UnsafeUtility.AlignOf<int>();
                Out->visibleInstances = (int*)UnsafeUtility.Malloc(totalVis * sizeof(int), viAlign, Allocator.TempJob);
            }
            else
            {
                Out->visibleInstances = null;
            }

            // drawCommands: allocate uncompressed; cross-chunk merge will compact later
            Out->drawCommandCount = totalCmd;
            if (totalCmd > 0)
            {
                int cmdAlign = UnsafeUtility.AlignOf<BatchDrawCommand>();
                Out->drawCommands = (BatchDrawCommand*)UnsafeUtility.Malloc(
                    totalCmd * UnsafeUtility.SizeOf<BatchDrawCommand>(), cmdAlign, Allocator.TempJob);
            }
            else
            {
                Out->drawCommands = null;
            }

       
            if (Out->drawRangeCount > 0)
            {
                Out->drawRanges[0].drawCommandsBegin = 0;
                Out->drawRanges[0].drawCommandsCount = (uint)totalCmd;
                Out->drawRanges[0].filterSettings = new BatchFilterSettings { renderingLayerMask = 0xffffffffu };
            }

            Out->drawCommandPickingInstanceIDs = null;
            Out->instanceSortingPositions = null;
            Out->instanceSortingPositionFloatCount = 0;
        }
    }

    [BurstCompile]
    public unsafe struct HISMEmitWriteJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<HISMChunk> Chunks;
        [ReadOnly] public NativeArray<ChunkRuntime> ChunkRuntimes;
        [ReadOnly] public NativeStream.Reader Reader;

        [ReadOnly] public NativeArray<int> CmdOffsets;   // exclusive
        [ReadOnly] public NativeArray<int> VisOffsets;   // exclusive
        [ReadOnly] public NativeArray<int> RunCounts;    // for bounds
        [ReadOnly] public NativeArray<int> VisibleCounts;

        [NativeDisableUnsafePtrRestriction] public BatchCullingOutputDrawCommands* Out;
        [NativeDisableUnsafePtrRestriction] public MeshMaterialNode* MeshMaterials;
        public int MeshMaterialsLength;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int ClampLod(in MeshMaterialIndexCount mm, int lod)
            => math.clamp(lod, 0, math.max(0, mm.count - 1));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void FlushIfHaveRun(BatchCullingOutputDrawCommands* Out, ref bool haveRun, ref BatchDrawCommand cur, ref int cmdW, int visW)
        {
            if (!haveRun) return;
            cur.visibleCount = (uint)(visW - (int)cur.visibleOffset);
            Out->drawCommands[cmdW++] = cur;
            haveRun = false;
        }

        public void Execute(int ci)
        {
            int cmdBase = CmdOffsets[ci];
            int visBase = VisOffsets[ci];

            var reader = Reader;
            int n = reader.BeginForEachIndex(ci);

            var chunk = Chunks[ci];
            var prims = (HISMPrimitive*)chunk.Primitives.GetUnsafePtr();

            var rt = ChunkRuntimes[ci];
            var batchID = new BatchID { value = (uint)rt.BatchIndex };

            // write cursors
            int cmdW = cmdBase;
            int visW = visBase;

            // building command state (keep cross-range merge context)
            bool haveRun = false;
            BatchDrawCommand cur = default;
            int curMMIdx = -1;
            int curBatchIdx = -1;

            for (int k = 0; k < n; ++k)
            {
                var v = reader.Read<VisibleCmd>();
                int end = v.first + v.count;

                for (int j = v.first; j < end; ++j)
                {
                    var mm = prims[j].meshMaterialLods;
                    int mmIdx = mm.index + ClampLod(mm, v.lod);

                    // flush on state change
                    if (!haveRun || mmIdx != curMMIdx || rt.BatchIndex != curBatchIdx)
                    {
                        FlushIfHaveRun(Out, ref haveRun, ref cur, ref cmdW, visW);

                        // start a new command
                        var m = MeshMaterials[mmIdx];
                        cur = default;
                        cur.batchID = batchID;
                        cur.materialID = m.batchMaterialID;
                        cur.meshID = m.batchMeshID;
                        cur.submeshIndex = 0;           // submeshIndex is always 0
                        cur.splitVisibilityMask = 0xff;
                        cur.flags = 0;
                        cur.sortingPosition = 0;
                        cur.visibleOffset = (uint)visW;

                        curMMIdx = mmIdx;
                        curBatchIdx = rt.BatchIndex;
                        haveRun = true;
                    }

                    // append this instance
                    Out->visibleInstances[visW++] = rt.ChunkOffsetInBatch + j;
                }
            }

            reader.EndForEachIndex();

            // flush the last one
            FlushIfHaveRun(Out, ref haveRun, ref cur, ref cmdW, visW);
        }
    }

    [BurstCompile]
    public unsafe struct HISMCompactDrawCommandsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction] public BatchCullingOutputDrawCommands* Out;

        public void Execute()
        {
            int n = Out->drawCommandCount;
            if (n <= 1) return;

            int w = 0; // write cursor
            var prev = Out->drawCommands[0];
            for (int i = 1; i < n; ++i)
            {
                var cur = Out->drawCommands[i];

                // merge condition: same state + adjacent visible ranges in visibleInstances (contiguous)
                bool same =
                    prev.batchID.value == cur.batchID.value &&
                    prev.meshID.value == cur.meshID.value &&
                    prev.materialID.value == cur.materialID.value &&
                    prev.submeshIndex == cur.submeshIndex &&
                    prev.flags == cur.flags &&
                    prev.splitVisibilityMask == cur.splitVisibilityMask &&
                    prev.sortingPosition == cur.sortingPosition;

                if (same && prev.visibleOffset + prev.visibleCount == cur.visibleOffset)
                {
                    prev.visibleCount += cur.visibleCount;
                }
                else
                {
                    Out->drawCommands[w++] = prev;
                    prev = cur;
                }
            }
            Out->drawCommands[w++] = prev;

            Out->drawCommandCount = w;
            if (Out->drawRangeCount > 0)
                Out->drawRanges[0].drawCommandsCount = (uint)w;
        }
    }
    #endregion

    internal struct BatchInfo
    {
        public HeapBlock GPUMemoryAllocation;
        public SmallBlockAllocator SubbatchAllocator;
        public int HeadSubBatch;
        public int GraphicsArchetypeIndex;

        public int NextSameArch;
        public int PrevSameArch;
    }

    unsafe class HISMSystem : IDisposable, IHISMBackend
    {
        private const int kMaxCbufferSize = 64 * 1024;
        internal static readonly bool UseConstantBuffers = BatchRendererGroup.BufferTarget == BatchBufferTarget.ConstantBuffer;
        internal static readonly int MaxBytesPerCBuffer = math.min(kMaxCbufferSize, SystemInfo.maxConstantBufferSize);

        const int kGPUBufferSizeInitial = 2 * 1024 * 1024;
        const int kInitialMaxBatchCount = 1 * 1024;

        private FixedSizeAllocator m_GPUPersistentAllocator;
        private SubBatchAllocator m_SubBatchAllocator;
        private NativeList<int> m_ArchHead;
        private NativeParallelHashSet<int> m_ExistingSubBatchIndices;
        private NativeList<BatchInfo> m_BatchInfos;
        private NativeParallelHashSet<int> m_ExistingBatchIndices;

        private IntrusiveUnitBucketAllocator<ChunkProperty> m_ChunkMetadataAllocator;

        private int m_PersistentInstanceDataSize;

        private BatchRendererGroup m_BatchRendererGroup;
        private ThreadedBatchContext m_ThreadedBatchContext;

        private NativeArray<float4> m_SystemMemoryBuffer;

        private GraphicsBuffer m_GPUPersistentInstanceData;
        private GraphicsBufferHandle m_GPUPersistentInstanceBufferHandle;

        private NativeList<HISMChunk> m_Chunks;
        private NativeList<HISMChunk> m_NewChunks;

        // ‘À–– ±”≥…‰
        private NativeList<ChunkRuntime> m_ChunkRuntime;
        private NativeList<int> m_ChunkSubBatchID;

        // æ‰±˙”≥…‰
        private NativeParallelHashMap<ulong, int> m_HandleToChunkIndex;
        private NativeList<ulong> m_ChunkHandle;
        private ulong m_NextHandle;

        // ‰÷»æ”≥…‰
        private HIMRRenderArray m_RenderArray;

        static NativeParallelMultiHashMap<int, MaterialPropertyType> s_NameIDToMaterialProperties;

        private int m_MaxBatchIdPlusOne;

        protected void OnCreate()
        {
            m_PersistentInstanceDataSize = kGPUBufferSizeInitial;

            m_BatchRendererGroup = new BatchRendererGroup(this.OnPerformCulling, IntPtr.Zero);
            m_BatchRendererGroup.SetEnabledViewTypes(new BatchCullingViewType[]
            {
                BatchCullingViewType.Camera,
                BatchCullingViewType.Light,
                BatchCullingViewType.Picking,
                BatchCullingViewType.SelectionOutline
            });
            m_ThreadedBatchContext = m_BatchRendererGroup.GetThreadedBatchContext();

            m_GPUPersistentAllocator = new FixedSizeAllocator(MaxBytesPerCBuffer, (int)m_PersistentInstanceDataSize / MaxBytesPerCBuffer);
            m_SubBatchAllocator = new SubBatchAllocator(kInitialMaxBatchCount * 4);

            if (UseConstantBuffers)
                m_GPUPersistentInstanceData = new GraphicsBuffer(GraphicsBuffer.Target.Constant, (int)m_PersistentInstanceDataSize / 16, 16);
            else
                m_GPUPersistentInstanceData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int)m_PersistentInstanceDataSize / 4, 4);

            m_GPUPersistentInstanceBufferHandle = m_GPUPersistentInstanceData.bufferHandle;

            // Recreate to ensure alignment with possible updated size values
            m_GPUPersistentAllocator = new FixedSizeAllocator(MaxBytesPerCBuffer, (int)m_PersistentInstanceDataSize / MaxBytesPerCBuffer);
            m_SubBatchAllocator = new SubBatchAllocator(kInitialMaxBatchCount * 4);

            m_ArchHead = new NativeList<int>(256, Allocator.Persistent);
            m_ArchHead.Resize(256, NativeArrayOptions.UninitializedMemory);
            for (int i = 0; i < 256; ++i) m_ArchHead[i] = -1;

            m_ChunkMetadataAllocator = new IntrusiveUnitBucketAllocator<ChunkProperty>(32, 256);

            m_ExistingSubBatchIndices = new NativeParallelHashSet<int>(128, Allocator.Persistent);
            m_ExistingBatchIndices = new NativeParallelHashSet<int>(128, Allocator.Persistent);

            m_SystemMemoryBuffer = new NativeArray<float4>((int)m_PersistentInstanceDataSize / 16, Allocator.Persistent, NativeArrayOptions.ClearMemory);

            s_NameIDToMaterialProperties = new NativeParallelMultiHashMap<int, MaterialPropertyType>(16, Allocator.Persistent);
            RegisterMaterialPropertyType<float4x4>("unity_ObjectToWorld");

            m_MaxBatchIdPlusOne = 0;

            m_BatchInfos = new NativeList<BatchInfo>(kInitialMaxBatchCount, Allocator.Persistent);
            m_BatchInfos.Resize(1, NativeArrayOptions.ClearMemory);

            m_Chunks = new NativeList<HISMChunk>(Allocator.Persistent);
            m_NewChunks = new NativeList<HISMChunk>(Allocator.Persistent);

            m_ChunkRuntime = new NativeList<ChunkRuntime>(Allocator.Persistent);
            m_ChunkSubBatchID = new NativeList<int>(Allocator.Persistent);

            m_HandleToChunkIndex = new NativeParallelHashMap<ulong, int>(128, Allocator.Persistent);
            m_ChunkHandle = new NativeList<ulong>(Allocator.Persistent);
            m_NextHandle = 1;

            m_RenderArray = new HIMRRenderArray();
            m_RenderArray.Init(m_BatchRendererGroup);
            m_RenderArray._meshMaterialList = new UnsafeList<MeshMaterialNode>(0, Allocator.Persistent);
            m_RenderArray._rendererHashMap = new NativeHashMap<uint, MeshMaterialIndexCount>(128, Allocator.Persistent);
        }

        public static void RegisterMaterialPropertyType<T>(string propertyName, short overrideTypeSizeGPU = -1) where T : struct
        {
            short typeSizeCPU = (short)UnsafeUtility.SizeOf<T>();
            if (overrideTypeSizeGPU == -1)
                overrideTypeSizeGPU = typeSizeCPU;

            int nameID = Shader.PropertyToID(propertyName);
            var materialPropertyType = new MaterialPropertyType
            {
                TypeIndex = TypeId<T>.Value,
                NameID = nameID,
                SizeBytesCPU = typeSizeCPU,
                SizeBytesGPU = overrideTypeSizeGPU,
            };

            s_NameIDToMaterialProperties.Add(nameID, materialPropertyType);
        }

        // IHISMBackend °™°™ º”‘ÿ/–∂‘ÿ
        public ulong OnChunkLoaded(ref HISMChunk chunk, IntPtr bufferBase, int byteSize)
        {
            if (!AddChunk(ref chunk))
                return 0UL;

            int newIndex = m_Chunks.Length - 1;
            ulong handle = m_NextHandle++;
            m_HandleToChunkIndex[handle] = newIndex;
            m_ChunkHandle.Add(handle);
            return handle;
        }

        public void OnChunkUnloaded(ulong handle)
        {
            if (!m_HandleToChunkIndex.TryGetValue(handle, out int idx))
                return;

            m_HandleToChunkIndex.Remove(handle);

            int last = m_Chunks.Length - 1;
            ulong lastHandle = last >= 0 ? m_ChunkHandle[last] : 0UL;

            RemoveChunk(idx);

            if (idx != last)
            {
                m_HandleToChunkIndex[lastHandle] = idx;
                m_ChunkHandle[idx] = lastHandle;
            }

            if (last >= 0)
                m_ChunkHandle.Resize(last, NativeArrayOptions.UninitializedMemory);
        }

        // Entry point: add a chunk (upload to GPU, assign batch, record runtime map)
        public bool AddChunk(ref HISMChunk chunk)
        {
            int graphicsArchetypeIndex = chunk.Archetype;
            int numInstances = chunk.Primitives.Length;

            var overrides = chunk.MaterialProperties.AsUnsafeList();
            int maxPerBatch = MaxsPerCBufferBatch(overrides);

            var (existingBatchIndex, reservedSub, offset) =
                TryFindAvailableBatchForArchetype(graphicsArchetypeIndex, numInstances, maxPerBatch);

            bool ok;
            int batchIndex, chunkOffsetInBatch, subBatchID;

            if (existingBatchIndex != -1)
            {
                ok = AddSubBatchToExistingBatch(ref chunk,
                                                existingBatchIndex,
                                                reservedSub,
                                                offset,
                                                numInstances,
                                                maxPerBatch,
                                                out chunkOffsetInBatch,
                                                out subBatchID);
                batchIndex = existingBatchIndex;
            }
            else
            {
                ok = AddNewBatch(ref chunk, out batchIndex, out chunkOffsetInBatch, out subBatchID);
            }

            if (!ok) return false;

            // Record to system list and runtime map (index-aligned)
            m_Chunks.Add(chunk);
            m_ChunkRuntime.Add(new ChunkRuntime
            {
                BatchIndex = batchIndex,
                ChunkOffsetInBatch = chunkOffsetInBatch
            });
            m_ChunkSubBatchID.Add(subBatchID);

            return true;
        }

        private JobHandle OnPerformCulling(BatchRendererGroup rendererGroup,
                                           BatchCullingContext cullingContext,
                                           BatchCullingOutput cullingOutput,
                                           IntPtr userContext)
        {
            if (!m_Chunks.IsCreated || m_Chunks.Length == 0)
                return default;

            var planePackets = FrustumPlanes.BuildSOAPlanePackets(cullingContext.cullingPlanes, Allocator.TempJob);
            var lodParams = LODGroupExtensions.CalculateLODParams(cullingContext.lodParameters);

            // thresholds (replace with actual LOD policy as needed)
            float4 lodThresholdSq = new float4(32f * 32f, 64f * 64f, 128f * 128f, float.MaxValue);

            var stream = new NativeStream(m_Chunks.Length, Allocator.TempJob);
            var writer = stream.AsWriter();

            // 1) parallel culling + LOD (writes NativeStream)
            var cullJob = new HISMCullAndLodJob
            {
                LODParams = lodParams,
                Packet4 = planePackets,
                Chunks = m_Chunks.AsArray(),
                Writer = writer,
                LodThresholdSq = lodThresholdSq
            }.Schedule(m_Chunks.Length, 1);

            int chunkCount = m_Chunks.Length;

            // 2) per-chunk command/visible counts (parallel)
            var runCounts = new NativeArray<int>(chunkCount, Allocator.TempJob);
            var visCounts = new NativeArray<int>(chunkCount, Allocator.TempJob);

            var countJob = new HISMCountRunsJob
            {
                Chunks = m_Chunks.AsArray(),
                ChunkRuntimes = m_ChunkRuntime.AsArray(),
                Reader = stream.AsReader(),
                RunCounts = runCounts,
                VisibleCounts = visCounts
            }.Schedule(chunkCount, 1, cullJob);

            // 3) exclusive prefix sums for per-chunk write offsets
            var cmdOffsets = new NativeArray<int>(chunkCount, Allocator.TempJob);
            var visOffsets = new NativeArray<int>(chunkCount, Allocator.TempJob);

            var scanJob = new HISMExclusiveScanJob
            {
                RunCounts = runCounts,
                VisibleCounts = visCounts,
                CmdOffsets = cmdOffsets,
                VisOffsets = visOffsets
            }.Schedule(countJob);

            // 4) allocate output arrays (Malloc in job; Unity frees TempJob allocations after render)
            unsafe
            {
                var outPtr = (BatchCullingOutputDrawCommands*)cullingOutput.drawCommands.GetUnsafePtr();

                var allocJob = new HISMAllocateOutputJob
                {
                    RunCounts = runCounts,
                    VisibleCounts = visCounts,
                    Out = outPtr
                }.Schedule(scanJob);

                // 5) parallel emit (intra-chunk merged; cross-chunk adjacent merge in step 6)
                var emitJob = new HISMEmitWriteJob
                {
                    Chunks = m_Chunks.AsArray(),
                    ChunkRuntimes = m_ChunkRuntime.AsArray(),
                    Reader = stream.AsReader(),
                    CmdOffsets = cmdOffsets,
                    VisOffsets = visOffsets,
                    RunCounts = runCounts,
                    VisibleCounts = visCounts,
                    Out = outPtr,
                    MeshMaterials = m_RenderArray._meshMaterialList.Ptr,
                    MeshMaterialsLength = m_RenderArray._meshMaterialList.Length
                }.Schedule(chunkCount, 1, allocJob);

                // 6) cross-chunk adjacent command merge (in-place compact)
                var compactJob = new HISMCompactDrawCommandsJob
                {
                    Out = outPtr
                }.Schedule(emitJob);

                // 7) dispose temporaries following the dependency chain
                var dep1 = planePackets.Dispose(compactJob);
                var dep2 = stream.Dispose(dep1);
                var dep3 = runCounts.Dispose(dep2);
                var dep4 = visCounts.Dispose(dep3);
                var dep5 = cmdOffsets.Dispose(dep4);
                var dep6 = visOffsets.Dispose(dep5);

                return dep6;
            }
        }

        protected void OnUpdate() { }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void LinkBatchAsHead(int arch, int batchIndex)
        {
            var infos = m_BatchInfos.GetUnsafePtr();
            ref var bi = ref UnsafeUtility.AsRef<BatchInfo>(infos + batchIndex);

            int oldHead = m_ArchHead[arch];
            bi.PrevSameArch = InvalidIndex;
            bi.NextSameArch = oldHead;

            if (oldHead != InvalidIndex)
                UnsafeUtility.AsRef<BatchInfo>(infos + oldHead).PrevSameArch = batchIndex;

            m_ArchHead[arch] = batchIndex;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void LinkSubBatchAsHead(int batchIndex, int subBatchIndex)
        {
            var pool = m_SubBatchAllocator.GetUnsafePtr();
            var sub = pool + subBatchIndex;
            var bi = m_BatchInfos.GetUnsafePtr() + batchIndex;

            sub->PrevID = SubBatchAllocator.InvalidBatchNumber;
            sub->NextID = bi->HeadSubBatch;

            if (bi->HeadSubBatch != SubBatchAllocator.InvalidBatchNumber)
                pool[bi->HeadSubBatch].PrevID = subBatchIndex;

            bi->HeadSubBatch = subBatchIndex;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UploadChunkStreamsToGPU_EqualSize(
            ref HISMChunk chunk,
            UnsafeList<MaterialPropertyType> overrides,
            NativeArray<int> streamBegin,
            int chunkOffsetInBatch)
        {
            int numInstances = chunk.Primitives.Length;
            var chunkData = chunk.data.AsUnsafeList();

            byte* sysPtr = (byte*)m_SystemMemoryBuffer.GetUnsafePtr();
            int srcOffset = 0;

            for (int k = 0; k < overrides.Length; ++k)
            {
                int elemSize = overrides[k].SizeBytesGPU; // == SizeBytesCPU
                int dstByte = streamBegin[k] + chunkOffsetInBatch * elemSize;
                int bytes = numInstances * elemSize;

                UnsafeUtility.MemCpy(sysPtr + dstByte, chunkData.Ptr + srcOffset, bytes);

                int dstF4 = dstByte / 16;
                int sizeF4 = (bytes + 15) / 16;
                m_GPUPersistentInstanceData.SetData(m_SystemMemoryBuffer, dstF4, dstF4, sizeF4);

                srcOffset += bytes;
            }
        }

        // Create new batch: returns batchIndex / chunkOffsetInBatch / subBatchIndex
        bool AddNewBatch(ref HISMChunk chunk, out int outBatchIndex, out int outChunkOffsetInBatch, out int outSubBatchIndex)
        {
            outBatchIndex = -1;
            outChunkOffsetInBatch = -1;
            outSubBatchIndex = -1;

            int graphicsArchetypeIndex = chunk.Archetype;
            int numInstances = chunk.Primitives.Length;

            var overrides = chunk.MaterialProperties.AsUnsafeList();
            int maxPerBatch = MaxsPerCBufferBatch(overrides);

            var overrideSizes = new NativeArray<int>(overrides.Length, Allocator.Temp);
            int numProperties = overrides.Length;
            int batchMetadata = numProperties;

            for (int i = 0; i < overrides.Length; ++i)
                overrideSizes[i] = NextAlignedBy16(overrides[i].SizeBytesGPU * maxPerBatch);

            BatchInfo batchInfo = default;
            batchInfo.HeadSubBatch = SubBatchAllocator.InvalidBatchNumber;
            batchInfo.NextSameArch = InvalidIndex;
            batchInfo.PrevSameArch = InvalidIndex;
            batchInfo.GPUMemoryAllocation = m_GPUPersistentAllocator.Allocate();
            if (batchInfo.GPUMemoryAllocation.Empty)
            {
                m_GPUPersistentAllocator.Resize(m_GPUPersistentAllocator.MaxBlockCount * 2);
                batchInfo.GPUMemoryAllocation = m_GPUPersistentAllocator.Allocate();

                if (batchInfo.GPUMemoryAllocation.Empty)
                {
                    Debug.LogError($"Out of memory in the Entities Graphics GPU instance data buffer after resize.");
                    overrideSizes.Dispose();
                    return false;
                }
            }
            batchInfo.SubbatchAllocator = new SmallBlockAllocator(maxPerBatch);

            int allocationBegin = (int)batchInfo.GPUMemoryAllocation.begin;

            uint bindOffset = UseConstantBuffers ? (uint)allocationBegin : 0;
            uint bindWindowSize = UseConstantBuffers ? (uint)MaxBytesPerCBuffer : 0;

            var overrideStreamBegin = new NativeArray<int>(overrides.Length, Allocator.Temp);
            overrideStreamBegin[0] = allocationBegin;
            for (int i = 1; i < numProperties; ++i)
                overrideStreamBegin[i] = overrideStreamBegin[i - 1] + overrideSizes[i - 1];

            var overrideMetadata = new NativeArray<MetadataValue>(numProperties, Allocator.Temp);
            for (int i = 0; i < numProperties; ++i)
            {
                int gpuAddress = overrideStreamBegin[i] - (int)bindOffset;
                overrideMetadata[i] = CreateMetadataValue(overrides[i].NameID, gpuAddress, true);
            }

            var batchID = m_ThreadedBatchContext.AddBatch(overrideMetadata, m_GPUPersistentInstanceBufferHandle, bindOffset, bindWindowSize);
            int batchIndex = (int)batchID.value;

            Assert.IsTrue(batchIndex != 0, "Failed to add new BatchRendererGroup batch.");
            AddBatchIndex(batchIndex);

            int subBatchID = m_SubBatchAllocator.Allocate();
            if (subBatchID == -1)
            {
                Debug.LogError("Out of sub-batch indices in SubBatchAllocator");
                overrideMetadata.Dispose();
                overrideSizes.Dispose();
                overrideStreamBegin.Dispose();
                return false;
            }

            int offsetInBatch = batchInfo.SubbatchAllocator.Allocate(numInstances);
            if (offsetInBatch == -1)
            {
                Debug.LogError($"Failed to allocate {numInstances} instances in sub-batch allocator");
                m_SubBatchAllocator.Dealloc(subBatchID);
                overrideMetadata.Dispose();
                overrideSizes.Dispose();
                overrideStreamBegin.Dispose();
                return false;
            }

            var subPool = m_SubBatchAllocator.GetUnsafePtr();
            var sub = subPool + subBatchID;
            sub->ChunkOffsetInBatch = new HeapBlock((ulong)offsetInBatch, (ulong)(offsetInBatch + numInstances));
            sub->BatchID = batchIndex;

            // insert SubBatch at head
            m_BatchInfos.Resize(math.max(m_BatchInfos.Length, batchIndex + 1), NativeArrayOptions.ClearMemory);
            m_BatchInfos[batchIndex] = batchInfo;
            LinkSubBatchAsHead(batchIndex, subBatchID);

            sub->ChunkMetadataAllocation = m_ChunkMetadataAllocator.Allocate((ulong)batchMetadata);
            if (sub->ChunkMetadataAllocation.Empty)
            {
                Debug.LogWarning($"Out of memory in the Entities Graphics chunk metadata buffer. Attempted to allocate {batchMetadata} elements.");
                m_SubBatchAllocator.Dealloc(subBatchID);
                overrideMetadata.Dispose();
                overrideSizes.Dispose();
                overrideStreamBegin.Dispose();
                return false;
            }

            AddSubBatchIndex(subBatchID);
            EnsureArchetypeIndex(graphicsArchetypeIndex);

            // link batch to archetype list head
            batchInfo = m_BatchInfos[batchIndex];
            batchInfo.GraphicsArchetypeIndex = graphicsArchetypeIndex;
            m_BatchInfos[batchIndex] = batchInfo;
            LinkBatchAsHead(graphicsArchetypeIndex, batchIndex);

            // ---- write ChunkProperty metadata ----
            int chunkMetadataBegin = (int)sub->ChunkMetadataAllocation.begin;
            int chunkOffsetInBatch = (int)sub->ChunkOffsetInBatch.begin;
            var chunkProperties = m_ChunkMetadataAllocator.m_Buffer;
            for (int j = 0; j < numProperties; ++j)
            {
                var o = overrides[j];
                chunkProperties[chunkMetadataBegin + j] = new ChunkProperty
                {
                    TypeIndex = o.TypeIndex,
                    GPUDataBegin = overrideStreamBegin[j] + chunkOffsetInBatch * o.SizeBytesGPU,
                    ValueSizeBytesCPU = o.SizeBytesCPU,
                    ValueSizeBytesGPU = o.SizeBytesGPU
                };
            }

            // ---- grow persistent buffer if needed ----
            var persistentBytes = (ulong)(m_GPUPersistentAllocator.MaxBlockCount * MaxBytesPerCBuffer);
            if (persistentBytes > (ulong)m_PersistentInstanceDataSize)
            {
                while ((ulong)m_PersistentInstanceDataSize < persistentBytes)
                    m_PersistentInstanceDataSize *= 2;

                GraphicsBuffer newBuffer = UseConstantBuffers
                    ? new GraphicsBuffer(GraphicsBuffer.Target.Constant, (int)m_PersistentInstanceDataSize / 16, 16)
                    : new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int)m_PersistentInstanceDataSize / 4, 4);

                newBuffer.SetData(m_SystemMemoryBuffer, 0, 0, m_SystemMemoryBuffer.Length);

                var newSystemBuffer = new NativeArray<float4>((int)m_PersistentInstanceDataSize / 16, Allocator.Persistent, NativeArrayOptions.ClearMemory);
                if (m_SystemMemoryBuffer.IsCreated)
                {
                    NativeArray<float4>.Copy(m_SystemMemoryBuffer, newSystemBuffer, m_SystemMemoryBuffer.Length);
                    m_SystemMemoryBuffer.Dispose();
                }
                m_SystemMemoryBuffer = newSystemBuffer;

                m_GPUPersistentInstanceBufferHandle = newBuffer.bufferHandle;
                UpdateBatchBufferHandles();

                m_GPUPersistentInstanceData?.Dispose();
                m_GPUPersistentInstanceData = newBuffer;
            }

            // ---- upload equal-size SoA data ----
            UploadChunkStreamsToGPU_EqualSize(ref chunk, overrides, overrideStreamBegin, chunkOffsetInBatch);

            outBatchIndex = batchIndex;
            outChunkOffsetInBatch = chunkOffsetInBatch;
            outSubBatchIndex = subBatchID;

            overrideMetadata.Dispose();
            overrideSizes.Dispose();
            overrideStreamBegin.Dispose();

            return true;
        }

        private void UpdateBatchBufferHandles()
        {
            var it = m_ExistingBatchIndices.GetEnumerator();
            while (it.MoveNext())
            {
                var b = it.Current;
                m_BatchRendererGroup.SetBatchBuffer(new BatchID { value = (uint)b }, m_GPUPersistentInstanceBufferHandle);
            }
        }

        // Reuse existing batch: returns outChunkOffsetInBatch and outSubBatchIndex
        private bool AddSubBatchToExistingBatch(ref HISMChunk chunk,
                                                int batchIndex,
                                                int subBatchIndex,
                                                int offset,
                                                int numInstances,
                                                int maxPerBatch,
                                                out int outChunkOffsetInBatch,
                                                out int outSubBatchIndex)
        {
            outChunkOffsetInBatch = -1;
            outSubBatchIndex = -1;

            int graphicsArchetypeIndex = chunk.Archetype;

            var overrides = chunk.MaterialProperties.AsUnsafeList();
            int numProperties = overrides.Length;
            int batchTotalChunkMetadata = numProperties;

            int maxEntitiesPerBatch = maxPerBatch;
            if (offset < 0 || numInstances <= 0 || offset + numInstances > maxEntitiesPerBatch)
            {
                m_SubBatchAllocator.Dealloc(subBatchIndex);
                return false;
            }

            var batchInfo = m_BatchInfos.GetUnsafePtr() + batchIndex;
            if (!batchInfo->SubbatchAllocator.IsCreated || batchInfo->GraphicsArchetypeIndex != graphicsArchetypeIndex)
            {
                m_SubBatchAllocator.Dealloc(subBatchIndex);
                return false;
            }

            // SubBatch basics
            var pool = m_SubBatchAllocator.GetUnsafePtr();
            var sub = pool + subBatchIndex;

            ulong offsetBegin = (ulong)offset;
            ulong offsetEnd = (ulong)(offset + numInstances);
            if (offsetEnd <= offsetBegin || offsetEnd > (ulong)maxEntitiesPerBatch)
            {
                m_SubBatchAllocator.Dealloc(subBatchIndex);
                return false;
            }

            sub->ChunkOffsetInBatch = new HeapBlock(offsetBegin, offsetEnd);
            sub->BatchID = batchIndex;

            // insert SubBatch at head
            LinkSubBatchAsHead(batchIndex, subBatchIndex);
            AddSubBatchIndex(subBatchIndex);

            // allocate Chunk metadata
            sub->ChunkMetadataAllocation = m_ChunkMetadataAllocator.Allocate((ulong)batchTotalChunkMetadata);
            if (sub->ChunkMetadataAllocation.Empty)
            {
                Debug.LogWarning($"Out of memory in chunk metadata buffer for sub-batch");
                // free subBatchIndex to avoid leak
                m_SubBatchAllocator.Dealloc(subBatchIndex);
                return false;
            }

            // compute SoA stream starts (same as new-batch)
            var overrideStreamBegin = new NativeArray<int>(overrides.Length, Allocator.Temp);
            int allocationBegin = (int)batchInfo->GPUMemoryAllocation.begin;
            overrideStreamBegin[0] = allocationBegin;
            for (int i = 1; i < numProperties; ++i)
            {
                int sizeBytesComponent = NextAlignedBy16(overrides[i - 1].SizeBytesGPU * maxEntitiesPerBatch);
                overrideStreamBegin[i] = overrideStreamBegin[i - 1] + sizeBytesComponent;
            }

            // ---- write ChunkProperty metadata ----
            int chunkMetadataBegin = (int)sub->ChunkMetadataAllocation.begin;
            int chunkOffsetInBatch = offset;

            var chunkProperties = m_ChunkMetadataAllocator.m_Buffer;
            for (int j = 0; j < numProperties; ++j)
            {
                var o = overrides[j];
                int gpuBegin = overrideStreamBegin[j] + chunkOffsetInBatch * o.SizeBytesGPU;

                chunkProperties[chunkMetadataBegin + j] = new ChunkProperty
                {
                    TypeIndex = o.TypeIndex,
                    GPUDataBegin = gpuBegin,
                    ValueSizeBytesCPU = o.SizeBytesCPU,
                    ValueSizeBytesGPU = o.SizeBytesGPU
                };
            }

            // ---- upload equal-size SoA data ----
            UploadChunkStreamsToGPU_EqualSize(ref chunk, overrides, overrideStreamBegin, chunkOffsetInBatch);

            outChunkOffsetInBatch = chunkOffsetInBatch;
            outSubBatchIndex = subBatchIndex;

            overrideStreamBegin.Dispose();

            return true;
        }

        private static int NextAlignedBy16(int size)
        {
            return ((size + 15) >> 4) << 4;
        }

        internal static MetadataValue CreateMetadataValue(int nameID, int gpuAddress, bool isOverridden)
        {
            const uint kPerInstanceDataBit = 0x80000000;

            return new MetadataValue
            {
                NameID = nameID,
                Value = (uint)gpuAddress | (isOverridden ? kPerInstanceDataBit : 0),
            };
        }

        private static int MaxsPerCBufferBatch(UnsafeList<MaterialPropertyType> materialProperties)
        {
            int fixedBytes = materialProperties.Length * 16;
            int bytesPerEntity = 0;

            for (int i = 0; i < materialProperties.Length; ++i)
                bytesPerEntity += materialProperties[i].SizeBytesGPU;

            int maxBytes = HISMSystem.MaxBytesPerCBuffer;
            int maxBytesForEntities = maxBytes - fixedBytes;

            return maxBytesForEntities / math.max(1, bytesPerEntity);
        }

        private (int batchIndex, int subBatchIndex, int offset) TryFindAvailableBatchForArchetype(
            int graphicsArchetypeIndex, int requiredInstances, int maxPerBatch)
        {
            if (requiredInstances <= 0 || requiredInstances > maxPerBatch)
                return (-1, -1, -1);

            int reservedSub = m_SubBatchAllocator.Allocate();
            if (reservedSub == -1)
                return (-1, -1, -1);

            int want = requiredInstances;
            if (want > maxPerBatch)
            {
                m_SubBatchAllocator.Dealloc(reservedSub);
                return (-1, -1, -1);
            }

            EnsureArchetypeIndex(graphicsArchetypeIndex);

            var infos = m_BatchInfos.GetUnsafePtr();

            // linear scan batches of the same archetype; keep it simple
            for (int b = m_ArchHead[graphicsArchetypeIndex]; b != -1; b = UnsafeUtility.AsRef<BatchInfo>(infos + b).NextSameArch)
            {
                BatchInfo* bi = (infos + b);

                if (bi->GraphicsArchetypeIndex == graphicsArchetypeIndex && bi->SubbatchAllocator.IsCreated)
                {
                    int off = bi->SubbatchAllocator.Allocate(want);
                    if (off != -1)
                    {
                        if (off + want > maxPerBatch)
                        {
                            bi->SubbatchAllocator.Deallocate(off, want);
                        }
                        else
                        {
                            return (b, reservedSub, off);
                        }
                    }
                }
            }

            m_SubBatchAllocator.Dealloc(reservedSub);
            return (-1, -1, -1);
        }

        private void EnsureHaveSpaceForNewBatch()
        {
            int currentCapacity = m_BatchInfos.Length;
            int neededCapacity = m_MaxBatchIdPlusOne;

            if (currentCapacity >= neededCapacity) return;

            var newCapacity = math.max(2 * neededCapacity, currentCapacity + 1);

            m_BatchInfos.Resize(newCapacity, NativeArrayOptions.ClearMemory);

            var ptr = m_BatchInfos.GetUnsafePtr();
            for (int id = currentCapacity; id < newCapacity; ++id)
            {
                ref var bi = ref UnsafeUtility.AsRef<BatchInfo>(ptr + id);
                bi = default;
                bi.GraphicsArchetypeIndex = InvalidIndex;
                bi.NextSameArch = InvalidIndex;
                bi.PrevSameArch = InvalidIndex;
                bi.HeadSubBatch = InvalidIndex;
            }
        }

        private void AddBatchIndex(int id)
        {
            Assert.IsTrue(!m_ExistingBatchIndices.Contains(id), "New batch ID already marked as used");
            m_ExistingBatchIndices.Add(id);
            if (id + 1 > m_MaxBatchIdPlusOne)
                m_MaxBatchIdPlusOne = id + 1;
            EnsureHaveSpaceForNewBatch();
        }

        private void RemoveBatchIndex(int id)
        {
            if (!m_ExistingBatchIndices.Contains(id))
                Assert.IsTrue(false, $"Attempted to release an unused id {id}");
            m_ExistingBatchIndices.Remove(id);
        }

        private void AddSubBatchIndex(int id)
        {
            Assert.IsTrue(!m_ExistingSubBatchIndices.Contains(id), "New SubBatch ID already marked as used");
            m_ExistingSubBatchIndices.Add(id);
        }

        private void RemoveSubBatch(int subBatchIndex)
        {
            if (subBatchIndex == SubBatchAllocator.InvalidBatchNumber)
                return;

            SubBatch* subBatchPool = m_SubBatchAllocator.GetUnsafePtr();
            SubBatch* subBatch = subBatchPool + subBatchIndex;

            if (subBatch->BatchID == SubBatchAllocator.InvalidBatchNumber &&
                subBatch->ChunkOffsetInBatch.Empty &&
                subBatch->ChunkMetadataAllocation.Empty)
            {
                return;
            }

            m_ExistingSubBatchIndices.Remove(subBatchIndex);

            var batchIndex = subBatch->BatchID;

            BatchInfo* batchInfo = m_BatchInfos.GetUnsafePtr() + batchIndex;

            if (!subBatch->ChunkOffsetInBatch.Empty)
            {
                batchInfo->SubbatchAllocator.Deallocate((int)subBatch->ChunkOffsetInBatch.begin, (int)subBatch->ChunkOffsetInBatch.Length);
            }
            if (!subBatch->ChunkMetadataAllocation.Empty)
            {
                var metadataAddress = subBatch->ChunkMetadataAllocation;
                var chunkProperties = m_ChunkMetadataAllocator.m_Buffer;
                for (int i = (int)metadataAddress.begin; i < (int)metadataAddress.end; i++)
                    chunkProperties[i] = default;

                m_ChunkMetadataAllocator.Free(ref subBatch->ChunkMetadataAllocation);
            }

            if (subBatch->PrevID != SubBatchAllocator.InvalidBatchNumber)
            {
                subBatchPool[subBatch->PrevID].NextID = subBatch->NextID;
            }
            else
            {
                batchInfo->HeadSubBatch = subBatch->NextID;
            }

            if (subBatch->NextID != SubBatchAllocator.InvalidBatchNumber)
            {
                subBatchPool[subBatch->NextID].PrevID = subBatch->PrevID;
            }

            m_SubBatchAllocator.Dealloc(subBatchIndex);

            if (batchInfo->HeadSubBatch == SubBatchAllocator.InvalidBatchNumber)
            {
                RemoveBatch(batchIndex);
            }
        }

        // O(1) remove a chunk (free SubBatch and metadata; keep arrays compact)
        // Note: call when not concurrent with OnPerformCulling (e.g., main thread between frames).
        public bool RemoveChunk(int chunkIndex)
        {
            if (!m_Chunks.IsCreated || chunkIndex < 0 || chunkIndex >= m_Chunks.Length)
                return false;

            int subBatchIndex = m_ChunkSubBatchID[chunkIndex];

            // free this chunk's SubBatch (reclaims GPU blocks, chunk metadata, list nodes; auto-removes batch if empty)
            RemoveSubBatch(subBatchIndex);

            // compact the three parallel arrays (swap-back)
            int last = m_Chunks.Length - 1;
            if (chunkIndex != last)
            {
                m_Chunks[chunkIndex] = m_Chunks[last];
                m_ChunkRuntime[chunkIndex] = m_ChunkRuntime[last];
                m_ChunkSubBatchID[chunkIndex] = m_ChunkSubBatchID[last];
            }

            // shrink lengths
            m_Chunks.Resize(last, NativeArrayOptions.UninitializedMemory);
            m_ChunkRuntime.Resize(last, NativeArrayOptions.UninitializedMemory);
            m_ChunkSubBatchID.Resize(last, NativeArrayOptions.UninitializedMemory);

            return true;
        }

        const int InvalidIndex = -1;

        private void RemoveBatch(int batchIndex)
        {
            ref var p = ref m_BatchInfos.GetUnsafePtr()[batchIndex];
            int oldArch = p.GraphicsArchetypeIndex;

            if (oldArch != InvalidIndex)
            {
                UnlinkBatchFromArchList(batchIndex, oldArch);
            }

            RemoveBatchIndex(batchIndex);

            if (!p.GPUMemoryAllocation.Empty)
            {
                m_GPUPersistentAllocator.Dealloc(p.GPUMemoryAllocation);
                p.GPUMemoryAllocation = default;
            }

            if (p.SubbatchAllocator.IsCreated)
            {
                p.SubbatchAllocator.Dispose();
            }

            p.GraphicsArchetypeIndex = InvalidIndex;

            m_ThreadedBatchContext.RemoveBatch(new BatchID { value = (uint)batchIndex });
        }

        private void EnsureArchetypeIndex(int arch)
        {
            if (arch < 0) return;
            if (!m_ArchHead.IsCreated)
            {
                m_ArchHead = new NativeList<int>(math.max(256, arch + 1), Allocator.Persistent);
                m_ArchHead.Resize(math.max(256, arch + 1), NativeArrayOptions.UninitializedMemory);
                for (int i = 0; i < m_ArchHead.Length; ++i) m_ArchHead[i] = -1;
                return;
            }
            if (m_ArchHead.Length <= arch)
            {
                int old = m_ArchHead.Length;
                m_ArchHead.Resize(arch + 1, NativeArrayOptions.UninitializedMemory);
                for (int i = old; i < m_ArchHead.Length; ++i) m_ArchHead[i] = -1;
            }
        }

        private void UnlinkBatchFromArchList(int batchIndex, int arch)
        {
            var batchInfos = m_BatchInfos.GetUnsafePtr();
            ref var bi = ref UnsafeUtility.AsRef<BatchInfo>(batchInfos + batchIndex);
            int prev = bi.PrevSameArch;
            int next = bi.NextSameArch;
            if (prev != -1) UnsafeUtility.AsRef<BatchInfo>(batchInfos + prev).NextSameArch = next;
            else m_ArchHead[arch] = next;
            if (next != -1) UnsafeUtility.AsRef<BatchInfo>(batchInfos + next).PrevSameArch = prev;
            bi.PrevSameArch = -1; bi.NextSameArch = -1;
        }

        public void Dispose()
        {
            if (m_Chunks.IsCreated) m_Chunks.Dispose();
            if (m_NewChunks.IsCreated) m_NewChunks.Dispose();
            if (m_ChunkRuntime.IsCreated) m_ChunkRuntime.Dispose();
            if (m_ChunkSubBatchID.IsCreated) m_ChunkSubBatchID.Dispose();

            if (m_ArchHead.IsCreated) m_ArchHead.Dispose();
            if (m_ExistingBatchIndices.IsCreated) m_ExistingBatchIndices.Dispose();
            if (m_ExistingSubBatchIndices.IsCreated) m_ExistingSubBatchIndices.Dispose();
            if (m_BatchInfos.IsCreated) m_BatchInfos.Dispose();

            m_BatchRendererGroup?.Dispose();
            m_GPUPersistentInstanceData?.Dispose();

            if (m_SystemMemoryBuffer.IsCreated) m_SystemMemoryBuffer.Dispose();

            m_ChunkMetadataAllocator.Dispose();

            
            if (m_RenderArray._rendererHashMap.IsCreated) m_RenderArray._rendererHashMap.Dispose();
            if (m_RenderArray._meshMaterialList.Ptr != null) m_RenderArray._meshMaterialList.Dispose();

            if (m_HandleToChunkIndex.IsCreated) m_HandleToChunkIndex.Dispose();
            if (m_ChunkHandle.IsCreated) m_ChunkHandle.Dispose();
        }
    }
}
