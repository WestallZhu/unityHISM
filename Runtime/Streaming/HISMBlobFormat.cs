
// HISMBlobFormat_SR.cs
// Self-Relative (SR) Blob format: zero-fixup loading using self-relative BlobArray<T>.

#nullable enable
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Renderloom.HISM.Streaming
{
    // -----------------------------
    // Self-Relative BlobArray<T>
    // -----------------------------
    public unsafe struct BlobArray<T> where T : unmanaged
    {
        internal int m_OffsetPtr; // relative to the address of this field
        internal int m_Length;
        public int Length => m_Length;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetUnsafePtr()
        {
            fixed (int* thisPtr = &m_OffsetPtr)
            {
                return (byte*)thisPtr + m_OffsetPtr;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UnsafeList<T> AsUnsafeList()
        {
            return new UnsafeList<T>((T*)GetUnsafePtr(), m_Length);
        }
    }

    // -----------------------------
    // Manifest (little-endian on disk)
    // -----------------------------
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct HISMWorldManifestHeader
    {
        public uint  Magic;        // 'HWMF' = 0x464D5748
        public ushort Version;     // 0x0001
        public ushort Flags;       // bit0: packFileMode?
        public float CellSize;     // e.g. 64
        public float3 Origin;      // grid origin
        public int   ArchetypeCount;
        public int   CellCount;
        public AABB  WorldBounds;
        public fixed byte Reserved[32];
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct HISMCellIndex
    {
        public int2  Coord;       // (ix, iz)
        public AABB  Bounds;      // cell bounds
        public int   Archetype;
        public long  FileOffset;  // offset inside pack (if pack mode)
        public int   ByteSize;    // total bytes of this SR-chunk
    }

    // -----------------------------
    // Attribute descriptor for SoA "data"
    // -----------------------------
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct HISMAttrDesc
    {
        public MaterialPropertyType Type;
        public int  ElementSize;     // bytes per element (4/8/16...)
        public int  Stride;          // usually == ElementSize
        public int  StreamOffset;    // offset within "data" region
        public int  Reserved;
    }

    // -----------------------------
    // Self-Relative HISM Chunk (disk == memory)
    // -----------------------------
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct HISMChunkMR
    {
        public uint Magic;       // 'HCSR' = 0x52534348
        public ushort Version;     // 0x0001
        public ushort Flags;
        public int Archetype;
        public AABB BoundingBox;

        public BlobArray<HISMNode> BVHTree;
        public BlobArray<MaterialPropertyType> MaterialProperties;
        public BlobArray<HISMPrimitive> Primitives;
        public BlobArray<byte> data;            // SoA attributes
        public BlobArray<HISMAttrDesc> AttrDescs;       // optional

        public BlobArray<int> MaterialIndices; // indices -> AssetTable.Materials
        public BlobArray<int> MeshIndices;     // indices -> AssetTable.Meshes

    }

    // -----------------------------
    // Utility for distances (XZ)
    // -----------------------------
    public static class AABBM
    {
        public static float DistanceXZ(in AABB bounds, float3 p)
        {
            var min = bounds.Min;
            var max = bounds.Max;
            float cx = math.clamp(p.x, min.x, max.x);
            float cz = math.clamp(p.z, min.z, max.z);
            float dx = p.x - cx;
            float dz = p.z - cz;
            return math.sqrt(dx * dx + dz * dz);
        }
    }

    // -----------------------------
    // ChunkKey for dictionaries
    // -----------------------------
    public readonly struct ChunkKey : IEquatable<ChunkKey>
    {
        public readonly int2 Coord;
        public readonly int Archetype;

        public ChunkKey(int2 coord, int archetype)
        {
            Coord = coord;
            Archetype = archetype;
        }
        public bool Equals(ChunkKey other) => Coord.x == other.Coord.x && Coord.y == other.Coord.y && Archetype == other.Archetype;
        public override bool Equals(object? obj) => obj is ChunkKey k && Equals(k);
        public override int GetHashCode() => (Coord.x * 73856093) ^ (Coord.y * 19349663) ^ (Archetype * 83492791);
        public override string ToString() => $"[{Coord.x},{Coord.y}]@{Archetype}";
    }
}
