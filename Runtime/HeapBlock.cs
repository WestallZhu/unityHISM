
using System;
namespace Renderloom.Rendering
{
    public struct HeapBlock : IComparable<HeapBlock>, IEquatable<HeapBlock>
    {
        /// <summary>
        /// The beginning of the allocated heap block.
        /// </summary>
        public ulong begin { get { return m_Begin; } }

        /// <summary>
        /// The end of the allocated heap block.
        /// </summary>
        public ulong end { get { return m_End; } }

        public bool IsValid => m_Begin != ulong.MaxValue && m_End > m_Begin;

        internal ulong m_Begin;
        internal ulong m_End;

        internal HeapBlock(ulong begin, ulong end)
        {
            m_Begin = begin;
            m_End = end;
        }

        /// <summary>
        /// Creates new HeapBlock that starts at the given index and is of given size.
        /// </summary>
        /// <param name="begin">The start index for the block.</param>
        /// <param name="size">The size of the block.</param>
        /// <returns>Returns a new instance of HeapBlock.</returns>
        internal static HeapBlock OfSize(ulong begin, ulong size)
        {
            return new HeapBlock(begin, begin + size);
        }

        /// <summary>
        /// The length of the HeapBlock.
        /// </summary>
        public ulong Length { get { return m_End - m_Begin; } }

        /// <summary>
        /// Indicates whether the HeapBlock is empty. This is true if the HeapBlock is empty and false otherwise.
        /// </summary>
        public bool Empty { get { return Length == 0; } }

        /// <inheritdoc/>
        public int CompareTo(HeapBlock other) { return m_Begin.CompareTo(other.m_Begin); }

        /// <inheritdoc/>
        public bool Equals(HeapBlock other) { return CompareTo(other) == 0; }
    }
}

