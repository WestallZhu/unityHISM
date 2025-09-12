
// IHISMBackend.cs
#nullable enable
using System;

namespace Renderloom.HISM.Streaming
{
    /// <summary>
    /// Backend interface for SR-chunks. IMPORTANT: 'in' passes by readonly-ref (no struct copy),
    /// so BlobArray self-relative offsets remain valid.
    /// </summary>
    public unsafe interface IHISMBackend : IDisposable
    {
        ulong OnChunkLoaded(ref  HISMChunk chunk, IntPtr bufferBase, int byteSize);
        void  OnChunkUnloaded(ulong handle);
    }

    public sealed class NullHISMBackend : IHISMBackend
    {
        public void Dispose() { }
        public ulong OnChunkLoaded(ref HISMChunk chunk, IntPtr bufferBase, int byteSize) => 0UL;
        public void OnChunkUnloaded(ulong handle) { }
    }
}
