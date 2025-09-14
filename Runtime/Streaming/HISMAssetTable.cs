
// HISMAssetTable_SR.cs
// Self-relative asset table: disk == memory, zero-fixup. Strings are UTF8 null-terminated in a single blob.

#nullable enable
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace Renderloom.HISM.Streaming
{
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct HISMAssetTable
    {
        public uint   Magic;       // 'HATS' = 0x53544148
        public ushort Version;     // 0x0001
        public ushort Flags;       // bit0: 0=Resources,1=Addressables
        public BlobArray<int>  MaterialNameOffsets; // offsets into StringBlob
        public BlobArray<int>  MeshNameOffsets;     // offsets into StringBlob
        public BlobArray<byte> StringBlob;          // UTF8 zero-terminated concatenation

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte* GetStringPtr(int offset) => (byte*)StringBlob.GetUnsafePtr() + offset;
    }

    /// <summary>
    /// Low-GC resolver:
    /// - Strings materialized once at LoadTable() to string[] (required by Unity APIs).
    /// - per-index caches are arrays (O(1) lookup, no hashing, no reallocation during runtime).
    /// - Fill-into APIs avoid List allocations; caller can reuse pooled lists.
    /// - Optional Addressables support via compile symbol USE_ADDRESSABLES.
    /// </summary>
    public unsafe sealed class HISMAssetResolver : IDisposable
    {
        IntPtr _buffer; int _bytes;
        HISMAssetTable* _table;

        // One-time materialized keys (strings are required by Unity's loading APIs)
        string[] _matKeys = Array.Empty<string>();
        string[] _meshKeys = Array.Empty<string>();

        // Per-index caches: arrays -> zero GC during lookup
        Material[] _matObjs = Array.Empty<Material>();
        Mesh[]     _meshObjs = Array.Empty<Mesh>();

#if USE_ADDRESSABLES
        System.Collections.Generic.Dictionary<int, UnityEngine.ResourceManagement.AsyncOperations.AsyncOperationHandle<Material>> _matHandles;
        System.Collections.Generic.Dictionary<int, UnityEngine.ResourceManagement.AsyncOperations.AsyncOperationHandle<Mesh>> _meshHandles;
        bool _useAddressables;
#else
        bool _useAddressables;
#endif

        public int MaterialCount => _matKeys.Length;
        public int MeshCount => _meshKeys.Length;

        public void LoadTable(string filePath)
        {
            // Managed read once, pin into native buffer to keep SR invariants if needed later.
            byte[] bytes = System.IO.File.ReadAllBytes(filePath);
            _buffer = (IntPtr)UnsafeUtility.Malloc(bytes.Length, 16, Allocator.Persistent);
            _bytes = bytes.Length;
            fixed (byte* src = bytes) UnsafeUtility.MemCpy((void*)_buffer, src, bytes.Length);
            _table = (HISMAssetTable*)_buffer;

            if (_table->Magic != 0x53544148u /*'HATS'*/ || _table->Version != 0x0001)
                throw new Exception("AssetTable header mismatch.");

            _useAddressables = (_table->Flags & 0x1) != 0;

            // Materialize keys once (no per-frame string allocs)
            var enc = Encoding.UTF8;
            var mats = _table->MaterialNameOffsets.AsUnsafeList();
            var meshs = _table->MeshNameOffsets.AsUnsafeList();
            _matKeys = new string[mats.Length];
            _meshKeys = new string[meshs.Length];

            for (int i = 0; i < mats.Length; i++) _matKeys[i] = ReadCString(enc, (byte*)_table->StringBlob.GetUnsafePtr(), mats.Ptr[i]);
            for (int i = 0; i < meshs.Length; i++) _meshKeys[i] = ReadCString(enc, (byte*)_table->StringBlob.GetUnsafePtr(), meshs.Ptr[i]);

            // Allocate object caches once
            _matObjs = new Material[_matKeys.Length];
            _meshObjs = new Mesh[_meshKeys.Length];

#if USE_ADDRESSABLES
            if (_useAddressables)
            {
                _matHandles = new(_matKeys.Length);
                _meshHandles = new(_meshKeys.Length);
            }
#endif
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static unsafe string ReadCString(Encoding enc, byte* basePtr, int offset)
        {
            byte* p = basePtr + offset;
            int len = 0;
            while (p[len] != 0) len++;
            return enc.GetString(p, len);
        }

        // ----------- Low-GC Resolve APIs -----------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Material ResolveMaterial(int index)
        {
            var obj = _matObjs[index];
            if ((object)obj != null) return obj;
            var key = _matKeys[index];
#if USE_ADDRESSABLES
            if (_useAddressables)
            {
                var h = UnityEngine.AddressableAssets.Addressables.LoadAssetAsync<Material>(key);
                h.WaitForCompletion();
                obj = h.Result;
                _matHandles[index] = h;
            }
            else
#endif
            {
                obj = Resources.Load<Material>(key);
            }
            _matObjs[index] = obj;
            return obj;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Mesh ResolveMesh(int index)
        {
            var obj = _meshObjs[index];
            if ((object)obj != null) return obj;
            var key = _meshKeys[index];
#if USE_ADDRESSABLES
            if (_useAddressables)
            {
                var h = UnityEngine.AddressableAssets.Addressables.LoadAssetAsync<Mesh>(key);
                h.WaitForCompletion();
                obj = h.Result;
                _meshHandles[index] = h;
            }
            else
#endif
            {
                obj = Resources.Load<Mesh>(key);
            }
            _meshObjs[index] = obj;
            return obj;
        }

        /// <summary>Fill into an existing list (cleared) to avoid GC. Capacity will be grown once if needed.</summary>
        public void ResolveMaterialsInto(UnsafeList<int> indices, List<Material> outList)
        {
            outList.Clear();
            if (outList.Capacity < indices.Length) outList.Capacity = indices.Length;
            for (int i = 0; i < indices.Length; i++) outList.Add(ResolveMaterial(indices[i]));
        }

        /// <summary>Fill into an existing list (cleared) to avoid GC. Capacity will be grown once if needed.</summary>
        public void ResolveMeshesInto(UnsafeList<int> indices, List<Mesh> outList)
        {
            outList.Clear();
            if (outList.Capacity < indices.Length) outList.Capacity = indices.Length;
            for (int i = 0; i < indices.Length; i++) outList.Add(ResolveMesh(indices[i]));
        }

        /// <summary>Optional: pre-load a set of assets into cache to avoid hitches.</summary>
        public void WarmupMaterials(NativeArray<int> indices)
        {
            for (int i = 0; i < indices.Length; i++) _ = ResolveMaterial(indices[i]);
        }
        public void WarmupMeshes(NativeArray<int> indices)
        {
            for (int i = 0; i < indices.Length; i++) _ = ResolveMesh(indices[i]);
        }

        /// <summary>Clear cached objects (does not unload Resources assets; for Addressables, releases handles).</summary>
        public void ReleaseAll()
        {
#if USE_ADDRESSABLES
            if (_useAddressables)
            {
                foreach (var kv in _matHandles) UnityEngine.AddressableAssets.Addressables.Release(kv.Value);
                foreach (var kv in _meshHandles) UnityEngine.AddressableAssets.Addressables.Release(kv.Value);
                _matHandles.Clear();
                _meshHandles.Clear();
            }
#endif
            Array.Clear(_matObjs, 0, _matObjs.Length);
            Array.Clear(_meshObjs, 0, _meshObjs.Length);
        }

        public void Dispose()
        {
            ReleaseAll();
            if (_buffer != IntPtr.Zero) { UnsafeUtility.Free((void*)_buffer, Allocator.Persistent); _buffer = IntPtr.Zero; }
        }
    }

    // ---------------- Simple per-type List Pool (chunk lifetime aware) ----------------
    internal static class HISMListPool<T> where T : class
    {
        static readonly Stack<List<T>> s_Pool = new Stack<List<T>>(64);

        public static List<T> Get(int minCapacity = 0)
        {
            lock (s_Pool)
            {
                if (s_Pool.Count > 0)
                {
                    var list = s_Pool.Pop();
                    list.Clear();
                    if (list.Capacity < minCapacity) list.Capacity = minCapacity;
                    return list;
                }
            }
            return new List<T>(minCapacity);
        }

        public static void Return(List<T> list)
        {
            list.Clear(); // we don't shrink capacity to keep amortized cost low
            lock (s_Pool) s_Pool.Push(list);
        }
    }
}
