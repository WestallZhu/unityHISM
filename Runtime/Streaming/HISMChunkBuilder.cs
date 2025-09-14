
// HISMChunkBuilder_SR_MR.cs
// Offline builder for SR chunk with per-chunk Material/Mesh indices.

#nullable enable
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Renderloom.HISM.Streaming
{
    public static unsafe class HISMChunkBuilder
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteRelInPlace<T>(ref BlobArray<T> field, void* dataPtr, int len) where T : unmanaged
        {
            fixed (int* pOff = &field.m_OffsetPtr) {
                field.m_Length = len;
                field.m_OffsetPtr = (int)((byte*)dataPtr - (byte*)pOff);
            }
        }

        public static byte[] BuildChunk<TBVH, TMat, TPrim>(
            int archetype, AABB bounds,
            TBVH[] bvh, TMat[] matProps, TPrim[] prims,
            byte[] dataBytes, HISMAttrDesc[] attrDescs,
            int[] materialIndices, int[] meshIndices,
            ushort version = 0x0001, ushort flags = 0)
            where TBVH  : unmanaged
            where TMat  : unmanaged
            where TPrim : unmanaged
        {
            int headerSize = Unsafe.SizeOf<HISMChunkMR>();
            int bvhBytes   = bvh.Length      * Unsafe.SizeOf<TBVH>();
            int mpBytes    = matProps.Length * Unsafe.SizeOf<TMat>();
            int prBytes    = prims.Length    * Unsafe.SizeOf<TPrim>();
            int dtBytes    = dataBytes.Length;
            int adBytes    = attrDescs.Length * Unsafe.SizeOf<HISMAttrDesc>();
            int miBytes    = materialIndices.Length * sizeof(int);
            int meBytes    = meshIndices.Length     * sizeof(int);

            int Align16(int x) => (x + 15) & ~15;

            int offBVH = Align16(headerSize);
            int offMP  = Align16(offBVH + bvhBytes);
            int offPR  = Align16(offMP  + mpBytes);
            int offDT  = Align16(offPR  + prBytes);
            int offAD  = Align16(offDT  + dtBytes);
            int offMI  = Align16(offAD  + adBytes);
            int offME  = Align16(offMI  + miBytes);
            int total  = Align16(offME  + meBytes);

            var buffer = new byte[total];
            fixed (byte* baseB = buffer)
            {
                var p = (HISMChunkMR*)baseB;
                p->Magic = 0x52534348u; // 'HCSR'
                p->Version = version;
                p->Flags = flags;
                p->Archetype = archetype;
                p->BoundingBox = bounds;

                if (bvhBytes   > 0) UnsafeUtility.MemCpy(baseB + offBVH, Unsafe.AsPointer(ref bvh[0]),       bvhBytes);
                if (mpBytes    > 0) UnsafeUtility.MemCpy(baseB + offMP,  Unsafe.AsPointer(ref matProps[0]), mpBytes);
                if (prBytes    > 0) UnsafeUtility.MemCpy(baseB + offPR,  Unsafe.AsPointer(ref prims[0]),    prBytes);
                if (dtBytes    > 0) UnsafeUtility.MemCpy(baseB + offDT,  Unsafe.AsPointer(ref dataBytes[0]), dtBytes);
                if (adBytes    > 0) UnsafeUtility.MemCpy(baseB + offAD,  Unsafe.AsPointer(ref attrDescs[0]), adBytes);

                fixed (int* mip = materialIndices) UnsafeUtility.MemCpy(baseB + offMI, mip, miBytes);
                fixed (int* mep = meshIndices)     UnsafeUtility.MemCpy(baseB + offME, mep, meBytes);

                WriteRelInPlace(ref p->BVHTree,            baseB + offBVH, bvh.Length);
                WriteRelInPlace(ref p->MaterialProperties, baseB + offMP,  matProps.Length);
                WriteRelInPlace(ref p->Primitives,         baseB + offPR,  prims.Length);
                WriteRelInPlace(ref p->data,               baseB + offDT,  dataBytes.Length);
                WriteRelInPlace(ref p->AttrDescs,          baseB + offAD,  attrDescs.Length);
                WriteRelInPlace(ref p->MaterialIndices,    baseB + offMI,  materialIndices.Length);
                WriteRelInPlace(ref p->MeshIndices,        baseB + offME,  meshIndices.Length);
            }
            return buffer;
        }
    }
}
