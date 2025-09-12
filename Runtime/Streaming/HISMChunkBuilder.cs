
// HISMChunkBuilder.cs
// Offline builder example: build a self-relative chunk blob (disk == memory).

#nullable enable
using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Renderloom.HISM.Streaming
{
    public static unsafe class HISMChunkBuilder
    {
        public struct BuildInput<TBVH, TMat, TPrim, TData, TAttr>
            where TBVH  : unmanaged
            where TMat  : unmanaged
            where TPrim : unmanaged
            where TData : unmanaged
            where TAttr : unmanaged
        {
            public TBVH[]  BVH;
            public TMat[]  MatProps;
            public TPrim[] Prims;
            public byte[]  Data;     // SoA packed
            public TAttr[] Attrs;    // attribute descriptors (HISMAttrDesc)
        }

        /// <summary>
        /// Build a self-relative chunk into a byte[] buffer.
        /// </summary>
        public static byte[] BuildChunk<TBVH, TMat, TPrim>(
            int archetype, AABB bounds,
            TBVH[] bvh, TMat[] matProps, TPrim[] prims,
            byte[] dataBytes, HISMAttrDesc[] attrDescs,
            ushort version = 0x0001, ushort flags = 0)
            where TBVH  : unmanaged
            where TMat  : unmanaged
            where TPrim : unmanaged
        {
            int headerSize = Unsafe.SizeOf<HISMChunk>();
            int bvhBytes   = bvh.Length      * Unsafe.SizeOf<TBVH>();
            int mpBytes    = matProps.Length * Unsafe.SizeOf<TMat>();
            int prBytes    = prims.Length    * Unsafe.SizeOf<TPrim>();
            int dtBytes    = dataBytes.Length;
            int adBytes    = attrDescs.Length * Unsafe.SizeOf<HISMAttrDesc>();

            // Optional: 16-byte align each section
            int Align16(int x) => (x + 15) & ~15;

            int offHeader = 0;
            int offBVH    = Align16(headerSize);
            int offMP     = Align16(offBVH + bvhBytes);
            int offPR     = Align16(offMP  + mpBytes);
            int offDT     = Align16(offPR  + prBytes);
            int offAD     = Align16(offDT  + dtBytes);
            int total     = Align16(offAD  + adBytes);

            var buffer = new byte[total];
            fixed (byte* baseB = buffer)
            {
                var pChunk = (HISMChunk*)baseB;
                // write static fields
                pChunk->Magic = 0x52534348u; // 'HCSR'
                pChunk->Version = version;
                pChunk->Flags = flags;
                pChunk->Archetype = archetype;
                pChunk->BoundingBox = bounds;

                // copy arrays
                UnsafeUtility.MemCpy(baseB + offBVH,  Unsafe.AsPointer(ref bvh[0]),  bvhBytes);
                UnsafeUtility.MemCpy(baseB + offMP,   Unsafe.AsPointer(ref matProps[0]), mpBytes);
                UnsafeUtility.MemCpy(baseB + offPR,   Unsafe.AsPointer(ref prims[0]), prBytes);
                UnsafeUtility.MemCpy(baseB + offDT,   Unsafe.AsPointer(ref dataBytes[0]), dtBytes);
                if (adBytes > 0)
                    UnsafeUtility.MemCpy(baseB + offAD, Unsafe.AsPointer(ref attrDescs[0]), adBytes);

                // write self-relative BlobArray fields
                WriteRelInPlace(ref pChunk->BVHTree,            baseB + offBVH, bvh.Length);
                WriteRelInPlace(ref pChunk->MaterialProperties, baseB + offMP,  matProps.Length);
                WriteRelInPlace(ref pChunk->Primitives,         baseB + offPR,  prims.Length);
                WriteRelInPlace(ref pChunk->data,               baseB + offDT,  dataBytes.Length);
                WriteRelInPlace(ref pChunk->AttrDescs,          baseB + offAD,  attrDescs.Length);
            }
            return buffer;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteRelInPlace<T>(ref BlobArray<T> field, void* dataPtr, int len) where T : unmanaged
        {
            fixed (int* pOff = &field.m_OffsetPtr)
            {
                field.m_Length = len;
                field.m_OffsetPtr = (int)((byte*)dataPtr - (byte*)pOff);
            }
        }
    }
}
