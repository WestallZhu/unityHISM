
// HISMStreamManager_SR_Job.cs
// Jobified + GC-free runtime (except file path string) for Self-Relative Blob chunks.
// Uses Burst jobs to build the target set and distance sort without managed collections.

#nullable enable
using Renderloom.HISM.Streaming;
using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.IO.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using static Renderloom.HISM.Streaming.AABBM;

namespace Renderloom.HISM.Streaming
{
    [BurstCompile]
    public static class KeyUtil
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong CoordKey(int ix, int iz)
        {
            // pack two uints into 64-bit (no hash collisions)
            return ((ulong)(uint)ix << 32) | (ulong)(uint)iz;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ChunkKey(int ix, int iz, int archetype)
        {
            // pack ix:iz:arch into 64 bits (ix: 22, iz: 22, arch: 20) — if out of range, wrap as uints (still unique within practical ranges)
            return ((ulong)(uint)ix << 42) | ((ulong)(uint)iz << 20) | (ulong)((uint)archetype & 0xFFFFF);
        }
    }

    internal enum ChunkState : byte { None, Queued, Reading, Loaded, Failed }

    [BurstCompile]
    internal struct GenerateCoordsJob : IJobParallelFor
    {
        public int MinX, MinZ, Width, Height;
        public NativeArray<ulong> OutCoordKeys; // CoordKey(ix,iz)

        public void Execute(int index)
        {
            int ix = MinX + (index % Width);
            int iz = MinZ + (index / Width);
            OutCoordKeys[index] = KeyUtil.CoordKey(ix, iz);
        }
    }

    [BurstCompile]
    internal struct ExpandCellsFromCoordsJob : IJobParallelFor
    {
        [ReadOnly] public NativeParallelMultiHashMap<ulong, int> GridToCellIndices; // key: CoordKey(ix,iz) -> many cell indices
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

    internal struct LoadCandidate : IComparable<LoadCandidate>
    {
        public ulong Key;   // packed (ix,iz,arch)
        public float Dist;  // distance to camera in XZ
        public int ByteSize;
        public int CompareTo(LoadCandidate other) => Dist.CompareTo(other.Dist);
    }

    [BurstCompile]
    internal struct ComputeDistancesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int> CellIndices;
        [ReadOnly] public NativeArray<HISMCellIndex> Cells;
        [ReadOnly] public float3 CameraPos;
        [ReadOnly] public float LoadDist;

        public NativeList<LoadCandidate>.ParallelWriter OutCandidates; // capacity >= CellIndices.Length

        public void Execute(int i)
        {
            int ci = CellIndices[i];
            var cell = Cells[ci];
            float d = AABBM.DistanceXZ(cell.Bounds, CameraPos);
            if (d <= LoadDist)
            {
                var cand = new LoadCandidate
                {
                    Key = KeyUtil.ChunkKey(cell.Coord.x, cell.Coord.y, cell.Archetype),
                    Dist = d,
                    ByteSize = cell.ByteSize
                };
                OutCandidates.AddNoResize(cand);
            }
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


    public unsafe sealed class HISMStreamManager : IDisposable
    {
        // Config
        public float CellSize = 64f;
        public float LoadDist = 256f;
        public float UnloadBias = 64f; // R_unload = LoadDist + UnloadBias
        public int MaxNewLoadsPerFrame = 4;
        public int MaxIOBytesPerFrame = 16 * 1024 * 1024;
        public ReadMode Mode = ReadMode.PackFile;
        public FixedString512Bytes ManifestPath;
        public FixedString512Bytes RootPath;
        public IHISMBackend Backend;

        public HISMWorldManifestHeader ManifestHeader { get; private set; }

        // Manifest data
        private NativeList<HISMCellIndex> _cells;                               // all cells
        private NativeParallelHashMap<ulong, int> _keyToCellIndex;              // (ix,iz,arch) -> index into _cells
        private NativeParallelMultiHashMap<ulong, int> _gridToCellIndices;      // CoordKey(ix,iz) -> many indices

        // Runtime state
        private NativeParallelHashMap<ulong, ChunkRuntime> _loaded;             // key -> runtime
        private NativeParallelHashMap<ulong, ChunkRuntime> _inflight;           // key -> runtime
        private NativeList<ulong> _tmpKeys;                                      // temp gather for removals

        // Scratch
        private NativeArray<ulong> _coordKeys;       // generated per frame
        private NativeQueue<int> _cellIdxQueue;    // expanded per frame
        private NativeList<int> _cellIdxList;     // consolidated
        private NativeList<LoadCandidate> _candidates; // filtered by distance

        public HISMStreamManager(IHISMBackend? backend = null, int initialCapacity = 4096)
        {
            Backend = backend ?? new NullHISMBackend();

            _cells = new NativeList<HISMCellIndex>(initialCapacity, Allocator.Persistent);
            _keyToCellIndex = new NativeParallelHashMap<ulong, int>(initialCapacity, Allocator.Persistent);
            _gridToCellIndices = new NativeParallelMultiHashMap<ulong, int>(initialCapacity, Allocator.Persistent);

            _loaded = new NativeParallelHashMap<ulong, ChunkRuntime>(initialCapacity, Allocator.Persistent);
            _inflight = new NativeParallelHashMap<ulong, ChunkRuntime>(initialCapacity / 4, Allocator.Persistent);
            _tmpKeys = new NativeList<ulong>(Allocator.Persistent);

            _coordKeys = default; // will be allocated per-frame sized
            _cellIdxQueue = new NativeQueue<int>(Allocator.Persistent);
            _cellIdxList = new NativeList<int>(Allocator.Persistent);
            _candidates = new NativeList<LoadCandidate>(Allocator.Persistent);
        }

        // ---------------- Manifest loading (minimal managed usage) ----------------
        public void LoadManifest(string manifestPath, string rootPath, ReadMode mode)
        {
            Mode = mode;
            ManifestPath = manifestPath;
            RootPath = rootPath;

            // Use managed FileStream once to read manifest; copy to NativeArray and parse (no GC allocations during runtime).
            // If you prefer, replace with AsyncReadManager to also avoid this managed stream.
            byte[] bytes = System.IO.File.ReadAllBytes(manifestPath);
            var na = new NativeArray<byte>(bytes.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            UnsafeUtility.MemCpy(na.GetUnsafePtr(), Unsafe.AsPointer(ref bytes[0]), bytes.Length);

            // Parse
            if (na.Length < Unsafe.SizeOf<HISMWorldManifestHeader>())
                throw new Exception("Manifest truncated.");
            ManifestHeader = UnsafeUtility.ReadArrayElement<HISMWorldManifestHeader>(na.GetUnsafePtr(), 0);

            if (ManifestHeader.Magic != 0x464D5748u /*'HWMF'*/ || ManifestHeader.Version != 0x0001)
                throw new Exception("Bad manifest header.");

            CellSize = ManifestHeader.CellSize;

            int off = Unsafe.SizeOf<HISMWorldManifestHeader>();
            int count = ManifestHeader.CellCount;
            int sz = Unsafe.SizeOf<HISMCellIndex>();
            if (na.Length < off + sz * count) throw new Exception("Manifest cells truncated.");

            _cells.Clear();
            _keyToCellIndex.Clear();
            _gridToCellIndices.Clear();

            _cells.ResizeUninitialized(count);
            void* dst = _cells.GetUnsafePtr();
            void* src = (byte*)na.GetUnsafePtr() + off;
            UnsafeUtility.MemCpy(dst, src, sz * count);

            // Build maps
            for (int i = 0; i < count; i++)
            {
                var ci = _cells[i];
                ulong key = KeyUtil.ChunkKey(ci.Coord.x, ci.Coord.y, ci.Archetype);
                _keyToCellIndex.TryAdd(key, i);
                ulong ckey = KeyUtil.CoordKey(ci.Coord.x, ci.Coord.y);
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
            int width = math.max(0, maxx - minx + 1);
            int height = math.max(0, maxz - minz + 1);
            int nCoords = width * height;

            if (_coordKeys.IsCreated) _coordKeys.Dispose();
            _coordKeys = new NativeArray<ulong>(nCoords, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            var genJob = new GenerateCoordsJob
            {
                MinX = minx,
                MinZ = minz,
                Width = width,
                Height = height,
                OutCoordKeys = _coordKeys
            };
            var genHandle = genJob.Schedule(nCoords, 64);

            // 2) Expand coords -> cell indices
            _cellIdxQueue.Clear();
            var expandJob = new ExpandCellsFromCoordsJob
            {
                GridToCellIndices = _gridToCellIndices,
                CoordKeys = _coordKeys,
                OutCellIndexQueue = _cellIdxQueue.AsParallelWriter()
            };
            var expandHandle = expandJob.Schedule(nCoords, 64, genHandle);

            // 3) Consolidate queue on main thread (no managed allocations)
            expandHandle.Complete();
            _cellIdxList.Clear();
            while (_cellIdxQueue.TryDequeue(out var ci))
                _cellIdxList.Add(ci);

            // 4) Distance filter in parallel
            _candidates.Clear();
            _candidates.Capacity = math.max(_candidates.Capacity, _cellIdxList.Length); // ensure AddNoResize safe

            var distJob = new ComputeDistancesJob
            {
                CellIndices = _cellIdxList.AsArray(),
                Cells = _cells.AsArray(),
                CameraPos = cameraPos,
                LoadDist = LoadDist,
                OutCandidates = _candidates.AsParallelWriter()
            };
            distJob.Schedule(_cellIdxList.Length, 64).Complete();

            // 5) Sort by distance (NativeSort) on main thread
            var candArr = new NativeArray<LoadCandidate>(_candidates.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            UnsafeUtility.MemCpy(candArr.GetUnsafePtr(), _candidates.GetUnsafePtr(), _candidates.Length * Unsafe.SizeOf<LoadCandidate>());
            Unity.Collections.NativeSortExtension.Sort(candArr);

            // 6) Schedule loads with budgets
            ScheduleLoadsFromSorted(candArr);
            candArr.Dispose();

            // 7) Poll & finalize
            PollReadsAndFinalize();

            // 8) Unloads
            ScheduleUnloads(cameraPos);

            // temp
            _coordKeys.Dispose();
        }

        private void ScheduleLoadsFromSorted(NativeArray<LoadCandidate> sorted)
        {
            int newLoads = 0;
            int bytesThisFrame = 0;

            for (int i = 0; i < sorted.Length; i++)
            {
                if (newLoads >= MaxNewLoadsPerFrame) break;

                var cand = sorted[i];
                if (_loaded.ContainsKey(cand.Key) || _inflight.ContainsKey(cand.Key))
                    continue;

                if (!_keyToCellIndex.TryGetValue(cand.Key, out int ci)) continue;
                var cell = _cells[ci];

                if (bytesThisFrame + cell.ByteSize > MaxIOBytesPerFrame) continue;

                // Allocate blob buffer
                IntPtr buffer = (IntPtr)UnsafeUtility.Malloc(cell.ByteSize, 64, Allocator.Persistent);
                if (buffer == IntPtr.Zero) { Debug.LogError("[HISM] Alloc failed"); continue; }

                // Path/offset
                string path;
                long offset;
                long size = cell.ByteSize;

                if (Mode == ReadMode.PackFile)
                {
                    path = RootPath.ToString();
                    offset = cell.FileOffset;
                }
                else
                {
                    path = System.IO.Path.Combine(RootPath.ToString(), cell.Archetype.ToString(), $"{cell.Coord.x}_{cell.Coord.y}.bin");
                    offset = 0;
                }

                var cmd = new ReadCommand { Buffer = (void*)buffer, Offset = offset, Size = size };
                var handle = AsyncReadManager.Read(path, &cmd, 1);

                var rt = new ChunkRuntime
                {
                    Key = cand.Key,
                    State = ChunkState.Reading,
                    BufferBase = buffer,
                    BufferBytes = cell.ByteSize,
                    ReadHandle = handle,
                    Chunk = null,
                    BackendHandle = 0UL
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

                    if (!ok)
                    {
                        UnsafeUtility.Free((void*)cr.BufferBase, Allocator.Persistent);
                        _tmpKeys.Add(key);
                        continue;
                    }

                    if (!TryGetSRChunkPointer(cr.BufferBase, cr.BufferBytes, out var pChunk))
                    {
                        UnsafeUtility.Free((void*)cr.BufferBase, Allocator.Persistent);
                        _tmpKeys.Add(key);
                        continue;
                    }

                    ulong h = Backend.OnChunkLoaded(ref  *pChunk, cr.BufferBase, cr.BufferBytes);

                    cr.State = ChunkState.Loaded;
                    cr.Chunk = pChunk;
                    cr.BackendHandle = h;

                    _loaded.TryAdd(key, cr);
                    _tmpKeys.Add(key);
                }
            }
            finally { kv.Dispose(); }

            for (int i = 0; i < _tmpKeys.Length; i++)
                _inflight.Remove(_tmpKeys[i]);
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

                    // lookup cell bounds
                    if (!_keyToCellIndex.TryGetValue(key, out int ci)) continue;
                    var cell = _cells[ci];

                    float d = DistanceXZ(cell.Bounds, cam);
                    if (d > runload)
                    {
                        Backend.OnChunkUnloaded(cr.BackendHandle);
                        UnsafeUtility.Free((void*)cr.BufferBase, Allocator.Persistent);
                        _tmpKeys.Add(key);
                    }
                }
            }
            finally { kv.Dispose(); }

            for (int i = 0; i < _tmpKeys.Length; i++)
                _loaded.Remove(_tmpKeys[i]);
        }

        public void Dispose()
        {
            // cancel inflight
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

            // unload loaded
            var lod = _loaded.GetKeyValueArrays(Allocator.Temp);
            for (int i = 0; i < lod.Length; i++)
            {
                Backend.OnChunkUnloaded(lod.Values[i].BackendHandle);
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

            Backend.Dispose();
        }

        // ---- SR helpers ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TryGetSRChunkPointer(IntPtr basePtr, int byteSize, out HISMChunk* pChunk)
        {
            pChunk = null;
            if (byteSize < UnsafeUtility.SizeOf<HISMChunk>()) return false;
            var hdr = *(HISMChunk*)basePtr;
            if (hdr.Magic != 0x52534348u /*'HCSR'*/ || hdr.Version != 0x0001) return false;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            // Optionally add range checks similar to previous file if needed
#endif
            pChunk = (HISMChunk*)basePtr;
            return true;
        }

        private unsafe struct ChunkRuntime
        {
            public ulong Key;
            public ChunkState State;
            public IntPtr BufferBase;
            public int BufferBytes;
            public ReadHandle ReadHandle;
            public HISMChunk* Chunk;
            public ulong BackendHandle;
        }
    }
}
