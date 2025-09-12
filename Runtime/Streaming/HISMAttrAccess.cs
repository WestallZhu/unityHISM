
// HISMAttrAccess.cs
#nullable enable
using System;
using Unity.Collections.LowLevel.Unsafe;

namespace Renderloom.HISM.Streaming
{
    public unsafe readonly struct AttrView
    {
        public readonly byte* Base;
        public readonly int   Stride;
        public readonly int   Offset;
        public readonly int   Count;

        public AttrView(byte* b, int stride, int offset, int count)
        { Base = b; Stride = stride; Offset = offset; Count = count; }

        public void* ElementPtr(int i) => Base + Offset + i * Stride;
    }

    public static unsafe class HISMAttrLookup
    {
        public static AttrView Get(in HISMChunk chunk, MaterialPropertyType type)
        {
            var descs = chunk.AttrDescs.AsUnsafeList();
            var data  = (byte*)chunk.data.GetUnsafePtr();
            for (int i = 0; i < descs.Length; i++)
            {
                if (descs.Ptr[i].Type.Equals(type))
                {
                    return new AttrView(data, descs.Ptr[i].Stride, descs.Ptr[i].StreamOffset, chunk.Primitives.Length);
                }
            }
            return default;
        }
    }
}
