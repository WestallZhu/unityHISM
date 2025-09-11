using UnityEngine.Assertions;
using System;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;

namespace Renderloom.Rendering
{
    /// <summary>
    /// Fixed-size block allocator for GPU memory management
    /// Provides efficient allocation and deallocation of fixed-size blocks
    /// </summary>
    public struct FixedSizeAllocator : IDisposable
    {
        /// <summary>Free block indices linked list</summary>
        NativeArray<int> m_BlockFreelist;
        /// <summary>Index of first free block</summary>
        int m_FirstFree;
        /// <summary>Number of free blocks available</summary>
        int m_FreeCount;
        /// <summary>Size of each block in bytes</summary>
        int m_BlockSize;
        
        /// <summary>
        /// Initialize fixed-size allocator
        /// </summary>
        /// <param name="blockSize">Size of each block in bytes</param>
        /// <param name="maxBlockCount">Maximum number of blocks</param>
        public FixedSizeAllocator(int blockSize, int maxBlockCount)
        {
            m_BlockFreelist = new NativeArray<int>(maxBlockCount, Allocator.Persistent);

            m_BlockSize = blockSize;

            m_FirstFree = 0;
            m_FreeCount = maxBlockCount;

            // Initialize free list as linked chain: 0->1->2->...->n-1->-1
            for (int i = 0; i < maxBlockCount; i++)
            {
                m_BlockFreelist[i] = i + 1;
            }
            m_BlockFreelist[maxBlockCount - 1] = -1; // Last block points to invalid
        }

        /// <summary>
        /// Allocate a fixed-size block
        /// </summary>
        /// <returns>HeapBlock representing the allocated block, empty HeapBlock if no space available</returns>
        public HeapBlock Allocate()
        {
            if (m_FreeCount == 0 || m_FirstFree < 0)
                return new HeapBlock(); // No free blocks available
                
            int idx = m_FirstFree;
            ulong memBeg = (ulong)(idx * m_BlockSize);
            HeapBlock block = new HeapBlock(memBeg, memBeg + (ulong)m_BlockSize);
            
            // Update free list: remove allocated block from chain
            m_FirstFree = m_BlockFreelist[idx];
            m_BlockFreelist[idx] = -2; // Mark as allocated
            --m_FreeCount;

            return block;
        }

        /// <summary>
        /// Deallocate a previously allocated block
        /// </summary>
        /// <param name="block">Block to deallocate</param>
        public void Dealloc(HeapBlock block)
        {
            if (block.Empty)
                return; // Nothing to deallocate

            // Calculate block index from memory address
            int blockIndex = (int)block.begin / m_BlockSize;
            
            // Add block back to front of free list
            m_BlockFreelist[blockIndex] = m_FirstFree;
            m_FirstFree = blockIndex;

            ++m_FreeCount;
        }


        /// <summary>True if all blocks are free (allocator is empty)</summary>
        public bool Empty { get { return m_FreeCount == m_BlockFreelist.Length; } }

        /// <summary>True if no blocks are free (allocator is full)</summary>
        public bool Full { get { return m_FreeCount == 0; } }
        
        /// <summary>Number of free blocks available for allocation</summary>
        public int FreeCount { get { return m_FreeCount; } }
        
        /// <summary>Maximum number of blocks this allocator can manage</summary>
        public int MaxBlockCount { get { return m_BlockFreelist.IsCreated ? m_BlockFreelist.Length : 0; } }
        
        /// <summary>Number of currently allocated blocks</summary>
        public int UsedCount { get { return MaxBlockCount - m_FreeCount; } }
        
        /// <summary>Ratio of used blocks to total blocks (0.0 to 1.0)</summary>
        public float UtilizationRatio { get { return MaxBlockCount > 0 ? (float)UsedCount / MaxBlockCount : 0f; } }


        /// <summary>
        /// Resizes the allocator to support more blocks
        /// </summary>
        /// <param name="newMaxBlockCount">New maximum block count (must be >= current count)</param>
        public void Resize(int newMaxBlockCount)
        {
            int currentMaxCount = m_BlockFreelist.Length;
            
            if (newMaxBlockCount <= currentMaxCount)
                return; // No need to resize if not growing
                
            Assert.IsTrue(newMaxBlockCount > currentMaxCount, 
                "New block count must be greater than current count");
            
            // Create new larger array
            var newBlockFreelist = new NativeArray<int>(newMaxBlockCount, Allocator.Persistent);
            
            // Copy existing data
            for (int i = 0; i < currentMaxCount; i++)
            {
                newBlockFreelist[i] = m_BlockFreelist[i];
            }
            
            // Initialize new blocks in the freelist
            // Link the new blocks to the existing free chain
            int lastNewBlock = newMaxBlockCount - 1;
            for (int i = currentMaxCount; i < newMaxBlockCount; i++)
            {
                newBlockFreelist[i] = (i < lastNewBlock) ? i + 1 : m_FirstFree;
            }
            
            // Update first free to point to the first new block
            if (m_FreeCount == 0)
            {
                // If we were completely full, start from first new block
                m_FirstFree = currentMaxCount;
            }
            else
            {
                // Insert new blocks at the beginning of the free chain for better allocation patterns
                newBlockFreelist[lastNewBlock] = m_FirstFree;
                m_FirstFree = currentMaxCount;
            }
            
            // Update free count
            m_FreeCount += (newMaxBlockCount - currentMaxCount);
            
            // Replace old array
            m_BlockFreelist.Dispose();
            m_BlockFreelist = newBlockFreelist;
        }

        /// <summary>
        /// Dispose allocator and release native memory
        /// </summary>
        public void Dispose()
        {
            if (m_BlockFreelist.IsCreated)
                m_BlockFreelist.Dispose();
        }
    }
}

