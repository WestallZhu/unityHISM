
// HISMStreamManager_SR_Job_MR_Bridge.cs
// Jobified + GC-free streaming manager (SR chunk) with Material/Mesh indices + AssetTable + HISMSystem bridge.
// Calls: ulong OnChunkLoaded(ref HISMChunk chunk, IntPtr bufferBase, int byteSize, List<Material> materials, List<Mesh> meshes)

#nullable enable
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.IO.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace Renderloom.HISM.Streaming
{
    public unsafe interface IHISMSystemLike
    {
        ulong OnChunkLoaded(ref HISMChunk chunk, IntPtr bufferBase, int byteSize, List<Material> materials, List<Mesh> meshes);
        void  OnChunkUnloaded(ulong handle);
    }

    // Runtime view for HISMSystem: UnsafeList<T> views (non-owning) into blob
    public unsafe struct HISMChunk
    {
        public int Archetype;
        public AABB BoundingBox;
        public UnsafeList<HISMNode>             BVHTree;
        public UnsafeList<MaterialPropertyType> MaterialProperties;
        public UnsafeList<HISMPrimitive>        Primitives;
        public UnsafeList<byte>                 data;
        public UnsafeList<HISMAttrDesc>         AttrDescs;
    }

    // ------------------ Key packing helpers (local to avoid conflicts) ------------------
    [BurstCompile]
    internal static class KeyUtilBridge
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong CoordKey(int ix, int iz) => ((ulong)(uint)ix << 32) | (ulong)(uint)iz;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ChunkKey(int ix, int iz, int archetype)
            => ((ulong)(uint)ix << 42) | ((ulong)(uint)iz << 20) | (ulong)((uint)archetype & 0xFFFFF);
    }

    // ------------------ Jobs ------------------
    [BurstCompile]
    internal struct GenerateCoordsJob_B : IJobParallelFor
    {
        public int MinX, MinZ, Width, Height;
        public NativeArray<ulong> OutCoordKeys; // CoordKey(ix,iz)

        public void Execute(int index)
        {
            int ix = MinX + (index % Width);
            int iz = MinZ + (index / Width);
            OutCoordKeys[index] = KeyUtilBridge.CoordKey(ix, iz);
        }
    }

    [BurstCompile]
    internal struct ExpandCellsFromCoordsJob_B : IJobParallelFor
    {
        [ReadOnly] public NativeParallelMultiHashMap<ulong, int> GridToCellIndices;
        [ReadOnly] public NativeArray<ulong> CoordKeys;
        public NativeQueue<int>.ParallelWriter OutCellIndexQueue;

        public void Execute(int i)
        {
            var key = CoordKeys[i];
            if (GridToCellIndices.TryGetFirstValue(key, out var idx, out var it))
            {
                do { OutCellIndexQueue.Enqueue(idx); }
                while (GridToCellIndices.TryGetNextValue(out idx, ref it));
            }
        }
    }

    internal struct LoadCandidate_B : IComparable<LoadCandidate_B>
    {
        public ulong Key;   // packed (ix,iz,arch)
        public float Dist;  // distance to camera in XZ
        public int   ByteSize;
        public int CompareTo(LoadCandidate_B other) => Dist.CompareTo(other.Dist);
    }

    [BurstCompile]
    internal struct ComputeDistancesJob_B : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int> CellIndices;
        [ReadOnly] public NativeArray<HISMCellIndex> Cells;
        [ReadOnly] public float3 CameraPos;
        [ReadOnly] public float LoadDist;
        public NativeList<LoadCandidate_B>.ParallelWriter OutCandidates;

        public void Execute(int i)
        {
            int ci = CellIndices[i];
            var cell = Cells[ci];
            float dx = math.max(cell.Bounds.Min.x - CameraPos.x, 0f) + math.max(CameraPos.x - cell.Bounds.Max.x, 0f);
            float dz = math.max(cell.Bounds.Min.z - CameraPos.z, 0f) + math.max(CameraPos.z - cell.Bounds.Max.z, 0f);
            float d = math.sqrt(dx*dx + dz*dz);
            if (d <= LoadDist) OutCandidates.AddNoResize(new LoadCandidate_B { Key = KeyUtilBridge.ChunkKey(cell.Coord.x, cell.Coord.y, cell.Archetype), Dist = d, ByteSize = cell.ByteSize });
        }
    }

    /// <summary>
    /// 读取模式：单Pack文件 or 分散bin（每Cell一份）
    /// </summary>
    public enum ReadMode
    {
        PackFile,  // 使用 Manifest 的 FileOffset/ByteSize 从单一大包读取
        PerCellBin // 路径: {Root}/{Archetype}/{ix}_{iz}.bin ；忽略 Manifest.FileOffset
    }

    internal enum ChunkState : byte { None, Queued, Reading, Loaded, Failed }

    // ------------------ Manager ------------------
    public unsafe sealed class HISMStreamManager : IDisposable
    {
        // Config
        public float CellSize = 64f;
        public float LoadDist = 256f;
        public float UnloadBias = 64f; // R_unload = LoadDist + UnloadBias
        public int   MaxNewLoadsPerFrame = 4;
        public int   MaxIOBytesPerFrame = 16 * 1024 * 1024;
        public ReadMode Mode = ReadMode.PackFile;
        public FixedString512Bytes ManifestPath;
        public FixedString512Bytes RootPath;

        public HISMWorldManifestHeader ManifestHeader { get; private set; }

        // Dependencies
        readonly IHISMSystemLike _backend;
        readonly HISMAssetResolver _resolver;
        readonly bool _disposeResolverOnDispose;

        // Manifest data
        private NativeList<HISMCellIndex> _cells;
        private NativeParallelHashMap<ulong, int> _keyToCellIndex;
        private NativeParallelMultiHashMap<ulong, int> _gridToCellIndices;

        // Runtime state
        private NativeParallelHashMap<ulong, ChunkRuntime> _loaded;
        private NativeParallelHashMap<ulong, ChunkRuntime> _inflight;
        private NativeList<ulong> _tmpKeys;

        // Scratch
        private NativeArray<ulong> _coordKeys;
        private NativeQueue<int>   _cellIdxQueue;
        private NativeList<int>    _cellIdxList;
        private NativeList<LoadCandidate_B> _candidates;

        public HISMStreamManager(IHISMSystemLike backend, HISMAssetResolver resolver, bool disposeResolverOnDispose = false, int initialCapacity = 4096)
        {
            _backend = backend;
            _resolver = resolver;
            _disposeResolverOnDispose = disposeResolverOnDispose;

            _cells     = new NativeList<HISMCellIndex>(initialCapacity, Allocator.Persistent);
            _keyToCellIndex    = new NativeParallelHashMap<ulong, int>(initialCapacity, Allocator.Persistent);
            _gridToCellIndices = new NativeParallelMultiHashMap<ulong, int>(initialCapacity, Allocator.Persistent);

            _loaded   = new NativeParallelHashMap<ulong, ChunkRuntime>(initialCapacity, Allocator.Persistent);
            _inflight = new NativeParallelHashMap<ulong, ChunkRuntime>(initialCapacity/4, Allocator.Persistent);
            _tmpKeys  = new NativeList<ulong>(Allocator.Persistent);

            _coordKeys     = default;
            _cellIdxQueue  = new NativeQueue<int>(Allocator.Persistent);
            _cellIdxList   = new NativeList<int>(Allocator.Persistent);
            _candidates    = new NativeList<LoadCandidate_B>(Allocator.Persistent);
        }

        // ---------------- Manifest loading ----------------
        public void LoadManifest(string manifestPath, string rootPath, ReadMode mode)
        {
            Mode = mode;
            ManifestPath = manifestPath;
            RootPath = rootPath;

            byte[] bytes = System.IO.File.ReadAllBytes(manifestPath);
            var na = new NativeArray<byte>(bytes.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            unsafe { fixed (byte* src = bytes) UnsafeUtility.MemCpy(na.GetUnsafePtr(), src, bytes.Length); }

            if (na.Length < UnsafeUtility.SizeOf<HISMWorldManifestHeader>()) throw new Exception("Manifest truncated.");
            unsafe { ManifestHeader = *(HISMWorldManifestHeader*)na.GetUnsafePtr(); }
            if (ManifestHeader.Magic != 0x464D5748u || ManifestHeader.Version != 0x0001) throw new Exception("Bad manifest header.");
            CellSize = ManifestHeader.CellSize;

            int off = UnsafeUtility.SizeOf<HISMWorldManifestHeader>();
            int count = ManifestHeader.CellCount;
            int sz = UnsafeUtility.SizeOf<HISMCellIndex>();
            if (na.Length < off + sz * count) throw new Exception("Manifest cells truncated.");

            _cells.Clear(); _keyToCellIndex.Clear(); _gridToCellIndices.Clear();
            _cells.ResizeUninitialized(count);
            unsafe {
                void* dst = _cells.GetUnsafePtr();
                void* src2 = (byte*)na.GetUnsafePtr() + off;
                UnsafeUtility.MemCpy(dst, src2, sz * count);
            }
            for (int i = 0; i < count; i++)
            {
                var ci = _cells[i];
                ulong key = KeyUtilBridge.ChunkKey(ci.Coord.x, ci.Coord.y, ci.Archetype);
                _keyToCellIndex.TryAdd(key, i);
                ulong ckey = KeyUtilBridge.CoordKey(ci.Coord.x, ci.Coord.y);
                _gridToCellIndices.Add(ckey, i);
            }

            na.Dispose();
        }

        // ---------------- Per-frame update ----------------
        public void UpdateStreaming(float3 cameraPos)
        {
            if (_cells.Length == 0) return;

            // 1) Generate coord keys in range
            int minx = (int)math.floor((cameraPos.x - LoadDist - ManifestHeader.Origin.x) / CellSize);
            int maxx = (int)math.floor((cameraPos.x + LoadDist - ManifestHeader.Origin.x) / CellSize);
            int minz = (int)math.floor((cameraPos.z - LoadDist - ManifestHeader.Origin.z) / CellSize);
            int maxz = (int)math.floor((cameraPos.z + LoadDist - ManifestHeader.Origin.z) / CellSize);
            int width  = math.max(0, maxx - minx + 1);
            int height = math.max(0, maxz - minz + 1);
            int nCoords = width * height;

            if (_coordKeys.IsCreated) _coordKeys.Dispose();
            _coordKeys = new NativeArray<ulong>(nCoords, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            var genJob = new GenerateCoordsJob_B { MinX = minx, MinZ = minz, Width = width, Height = height, OutCoordKeys = _coordKeys };
            var genHandle = genJob.Schedule(nCoords, 64);

            // 2) Expand to cell indices
            _cellIdxQueue.Clear();
            var expandJob = new ExpandCellsFromCoordsJob_B { GridToCellIndices = _gridToCellIndices, CoordKeys = _coordKeys, OutCellIndexQueue = _cellIdxQueue.AsParallelWriter() };
            var expandHandle = expandJob.Schedule(nCoords, 64, genHandle);

            // 3) Consolidate queue
            expandHandle.Complete();
            _cellIdxList.Clear();
            while (_cellIdxQueue.TryDequeue(out var ci))
                _cellIdxList.Add(ci);

            // 4) Distance filtering (parallel)
            _candidates.Clear();
            _candidates.Capacity = math.max(_candidates.Capacity, _cellIdxList.Length);
            var distJob = new ComputeDistancesJob_B {
                CellIndices = _cellIdxList.AsArray(),
                Cells = _cells.AsArray(),
                CameraPos = cameraPos,
                LoadDist = LoadDist,
                OutCandidates = _candidates.AsParallelWriter()
            };
            distJob.Schedule(_cellIdxList.Length, 64).Complete();

            // 5) Sort by distance
            var candArr = new NativeArray<LoadCandidate_B>(_candidates.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            UnsafeUtility.MemCpy(candArr.GetUnsafePtr(), _candidates.GetUnsafePtr(), _candidates.Length * UnsafeUtility.SizeOf<LoadCandidate_B>());
            Unity.Collections.NativeSortExtension.Sort(candArr);

            // 6) IO schedule
            ScheduleLoadsFromSorted(candArr);
            candArr.Dispose();

            // 7) finalize
            PollReadsAndFinalize();

            // 8) unload
            ScheduleUnloads(cameraPos);

            _coordKeys.Dispose();
        }

        private void ScheduleLoadsFromSorted(NativeArray<LoadCandidate_B> sorted)
        {
            int newLoads = 0;
            int bytesThisFrame = 0;

            for (int i = 0; i < sorted.Length; i++)
            {
                if (newLoads >= MaxNewLoadsPerFrame) break;
                var cand = sorted[i];
                if (_loaded.ContainsKey(cand.Key) || _inflight.ContainsKey(cand.Key)) continue;
                if (!_keyToCellIndex.TryGetValue(cand.Key, out int ci)) continue;

                var cell = _cells[ci];
                if (bytesThisFrame + cell.ByteSize > MaxIOBytesPerFrame) continue;

                IntPtr buffer = (IntPtr)UnsafeUtility.Malloc(cell.ByteSize, 64, Allocator.Persistent);
                if (buffer == IntPtr.Zero) { Debug.LogError("[HISM] Alloc failed"); continue; }

                string path; long offset; long size = (long)cell.ByteSize;
                if (Mode == ReadMode.PackFile) { path = RootPath.ToString(); offset = cell.FileOffset; }
                else { path = System.IO.Path.Combine(RootPath.ToString(), cell.Archetype.ToString(), $"{cell.Coord.x}_{cell.Coord.y}.bin"); offset = 0; }

                var cmd = new ReadCommand { Buffer = (void*)buffer, Offset = offset, Size = size };
                var handle = AsyncReadManager.Read(path, &cmd, 1);

                var rt = new ChunkRuntime {
                    Key = cand.Key, State = ChunkState.Reading, BufferBase = buffer, BufferBytes = cell.ByteSize, ReadHandle = handle,
                    Chunk = null, BackendHandle = 0UL
                };
                _inflight.TryAdd(cand.Key, rt);
                newLoads++;
                bytesThisFrame += cell.ByteSize;
            }
        }

        private void PollReadsAndFinalize()
        {
            if (_inflight.Count() == 0) return;
            _tmpKeys.Clear();

            var kv = _inflight.GetKeyValueArrays(Allocator.Temp);
            try
            {
                for (int i = 0; i < kv.Length; i++)
                {
                    ulong key = kv.Keys[i];
                    var cr = kv.Values[i];
                    if (cr.ReadHandle.Status == ReadStatus.InProgress) continue;

                    bool ok = cr.ReadHandle.Status == ReadStatus.Complete;
                    cr.ReadHandle.Dispose();

                    if (!ok) { UnsafeUtility.Free((void*)cr.BufferBase, Allocator.Persistent); _tmpKeys.Add(key); continue; }

                    // Validate SR MR header
                    if (!TryGetSRChunkMRPointer(cr.BufferBase, cr.BufferBytes, out var pChunk))
                    {
                        UnsafeUtility.Free((void*)cr.BufferBase, Allocator.Persistent);
                        _tmpKeys.Add(key);
                        continue;
                    }

                    // Build runtime view (UnsafeList)
                    var runtime = new HISMChunk {
                        Archetype = pChunk->Archetype,
                        BoundingBox = pChunk->BoundingBox,
                        BVHTree = pChunk->BVHTree.AsUnsafeList(),
                        MaterialProperties = pChunk->MaterialProperties.AsUnsafeList(),
                        Primitives = pChunk->Primitives.AsUnsafeList(),
                        data = pChunk->data.AsUnsafeList(),
                        AttrDescs = pChunk->AttrDescs.AsUnsafeList()
                    };

                    // Pooled lists to reduce GC; keep them until unload
                    var matIdxs = pChunk->MaterialIndices.AsUnsafeList();
                    var meshIdxs = pChunk->MeshIndices.AsUnsafeList();
                    var mats = HISMListPool<Material>.Get(matIdxs.Length);
                    var meshs = HISMListPool<Mesh>.Get(meshIdxs.Length);
                    _resolver.ResolveMaterialsInto(matIdxs, mats);
                    _resolver.ResolveMeshesInto(meshIdxs, meshs);

                    // Register
                    ulong h = _backend.OnChunkLoaded(ref runtime, cr.BufferBase, cr.BufferBytes, mats, meshs);

                    cr.State = ChunkState.Loaded;
                    cr.Chunk = pChunk;
                    cr.BackendHandle = h;

                    _loaded.TryAdd(key, cr);
                    _tmpKeys.Add(key);
                }
            }
            finally { kv.Dispose(); }

            for (int i = 0; i < _tmpKeys.Length; i++) _inflight.Remove(_tmpKeys[i]);
        }

        private void ScheduleUnloads(float3 cam)
        {
            if (_loaded.Count() == 0) return;
            _tmpKeys.Clear();
            float runload = LoadDist + UnloadBias;

            var kv = _loaded.GetKeyValueArrays(Allocator.Temp);
            try
            {
                for (int i = 0; i < kv.Length; i++)
                {
                    ulong key = kv.Keys[i];
                    var cr = kv.Values[i];
                    if (!_keyToCellIndex.TryGetValue(key, out int ci)) continue;
                    var cell = _cells[ci];
                    float dx = math.max(cell.Bounds.Min.x - cam.x, 0f) + math.max(cam.x - cell.Bounds.Max.x, 0f);
                    float dz = math.max(cell.Bounds.Min.z - cam.z, 0f) + math.max(cam.z - cell.Bounds.Max.z, 0f);
                    float d = math.sqrt(dx*dx + dz*dz);
                    if (d > runload)
                    {
                        _backend.OnChunkUnloaded(cr.BackendHandle);
                        UnsafeUtility.Free((void*)cr.BufferBase, Allocator.Persistent);
                        _tmpKeys.Add(key);
                    }
                }
            }
            finally { kv.Dispose(); }

            for (int i = 0; i < _tmpKeys.Length; i++) _loaded.Remove(_tmpKeys[i]);
        }

        public void Dispose()
        {
            var infl = _inflight.GetKeyValueArrays(Allocator.Temp);
            for (int i = 0; i < infl.Length; i++)
            {
                var cr = infl.Values[i];
                try { cr.ReadHandle.Dispose(); } catch { }
                UnsafeUtility.Free((void*)cr.BufferBase, Allocator.Persistent);
            }
            infl.Dispose();
            _inflight.Clear();
            _inflight.Dispose();

            var lod = _loaded.GetKeyValueArrays(Allocator.Temp);
            for (int i = 0; i < lod.Length; i++)
            {
                _backend.OnChunkUnloaded(lod.Values[i].BackendHandle);
                UnsafeUtility.Free((void*)lod.Values[i].BufferBase, Allocator.Persistent);
            }
            lod.Dispose();
            _loaded.Clear();
            _loaded.Dispose();

            _tmpKeys.Dispose();
            _candidates.Dispose();
            if (_coordKeys.IsCreated) _coordKeys.Dispose();
            _cellIdxQueue.Dispose();
            _cellIdxList.Dispose();

            _gridToCellIndices.Dispose();
            _keyToCellIndex.Dispose();
            _cells.Dispose();

            if (_disposeResolverOnDispose) _resolver.Dispose();
        }

        // ---- Helpers ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TryGetSRChunkMRPointer(IntPtr basePtr, int byteSize, out HISMChunkMR* p)
        {
            p = null;
            if (byteSize < UnsafeUtility.SizeOf<HISMChunkMR>()) return false;
            var hdr = *(HISMChunkMR*)basePtr;
            if (hdr.Magic != 0x52534348u /*'HCSR'*/ || hdr.Version != 0x0001) return false;
            p = (HISMChunkMR*)basePtr;
            return true;
        }

        private unsafe struct ChunkRuntime
        {
            public ulong Key;
            public ChunkState State;
            public IntPtr BufferBase;
            public int    BufferBytes;
            public ReadHandle ReadHandle;
            public HISMChunkMR* Chunk;
            public ulong BackendHandle;
        }
    }
}
