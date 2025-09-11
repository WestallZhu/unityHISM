using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
using UnityEngine;
#endif

namespace Renderloom.Rendering
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct FreeBlock
    {
        public int offset;   // starting instance index
        public int count;    // number of instances in block
        public int nextID;   // next free block node ID
        public int prevID;   // prev free block node ID
    }

    /// <summary>
    /// Pool of FreeBlock nodes. Supports growth with stable IDs.
    /// </summary>
    internal unsafe struct FreeBlockAllocator : IDisposable
    {
        private int m_Length;            // capacity (# of FreeBlock nodes)
        internal FreeBlock* m_Blocks;    // backing buffer
        private int m_FirstFree;         // head of free node list (node pool)
        private int m_FreeCount;         // number of nodes available in pool
        private const int InvalidID = -1;

        public FreeBlockAllocator(int capacity)
        {
            m_Length = capacity;
            long size = (long)UnsafeUtility.SizeOf<FreeBlock>() * capacity;
            m_Blocks = (FreeBlock*)UnsafeUtility.Malloc(size, UnsafeUtility.AlignOf<FreeBlock>(), Allocator.Persistent);
            UnsafeUtility.MemClear(m_Blocks, size);

            m_FirstFree = 0;
            m_FreeCount = capacity;

            // Initialize node pool free list
            for (int i = 0; i < capacity; i++)
            {
                m_Blocks[i].nextID = (i < capacity - 1) ? (i + 1) : InvalidID;
                m_Blocks[i].prevID = InvalidID; // prevID unused for node pool links
            }
        }

        public void Dispose()
        {
            if (m_Blocks != null)
            {
                UnsafeUtility.Free(m_Blocks, Allocator.Persistent);
                m_Blocks = null;
            }
            m_Length = 0;
            m_FirstFree = InvalidID;
            m_FreeCount = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int AllocateNode()
        {
            if (m_FreeCount == 0 || m_FirstFree == InvalidID)
                return InvalidID;

            int id = m_FirstFree;
            ref FreeBlock node = ref m_Blocks[id];
            m_FirstFree = node.nextID;
            node.nextID = InvalidID;
            node.prevID = InvalidID;
            m_FreeCount--;
            return id;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void FreeNode(int id)
        {
            ref FreeBlock node = ref m_Blocks[id];
            node.offset = 0;
            node.count = 0;
            node.prevID = InvalidID;
            node.nextID = m_FirstFree;
            m_FirstFree = id;
            m_FreeCount++;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FreeBlock* GetUnsafePtr() => m_Blocks;

        /// <summary>
        /// Ensure at least one free node is available; grow pool 2× if needed.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool EnsureFreeNode()
        {
            if (m_FreeCount > 0 && m_FirstFree != InvalidID) return true;
            int newCapacity = (m_Length > 0) ? (m_Length * 2) : 64;
            return Grow(newCapacity);
        }

        private bool Grow(int newCapacity)
        {
            if (newCapacity <= m_Length) return true;

            long oldSize = (long)UnsafeUtility.SizeOf<FreeBlock>() * m_Length;
            long newSize = (long)UnsafeUtility.SizeOf<FreeBlock>() * newCapacity;
            var newBlocks = (FreeBlock*)UnsafeUtility.Malloc(newSize, UnsafeUtility.AlignOf<FreeBlock>(), Allocator.Persistent);
            if (newBlocks == null)
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Debug.LogError("FreeBlockAllocator.Grow: allocation failed.");
#endif
                return false;
            }

            UnsafeUtility.MemClear(newBlocks, newSize);

            // Copy existing nodes (IDs remain stable)
            if (m_Blocks != null && m_Length > 0)
                UnsafeUtility.MemCpy(newBlocks, m_Blocks, oldSize);

            // Chain the added segment into the node pool
            int firstAdded = m_Length;
            for (int i = firstAdded; i < newCapacity; ++i)
            {
                newBlocks[i].nextID = (i < newCapacity - 1) ? (i + 1) : InvalidID;
                newBlocks[i].prevID = InvalidID;
            }
            int oldFirst = m_FirstFree;
            m_FirstFree = firstAdded;
            if (newCapacity - 1 >= 0)
                newBlocks[newCapacity - 1].nextID = oldFirst;

            m_FreeCount += (newCapacity - m_Length);

            // Swap buffers
            if (m_Blocks != null)
                UnsafeUtility.Free(m_Blocks, Allocator.Persistent);

            m_Blocks = newBlocks;
            m_Length = newCapacity;
            return true;
        }
    }

    /// <summary>
    /// Next-fit allocator for integer ranges [offset, offset+count),
    /// with doubly-linked free list, failure caching, and an ultra-light cursor cache for Deallocate.
    /// </summary>
    internal unsafe struct SmallBlockAllocator : IDisposable
    {
        private readonly int maxInstances;   // maximum addressable instances
        private FreeBlockAllocator blockPool;
        private int freeListHeadID;          // head of free ranges list
        private int allocCursorID;           // next-fit cursor
        private int maxFreeBlock;            // cached max block after a failed scan
        private bool maxDirty;               // true if structure changed since last fail
        private int totalFree;               // quick lower bound

        private const int InvalidID = -1;

        // ---- Tiny direct-mapped cursor cache (no bucketShift) ----
        // index = Mix32(offset) & CursorMask; equality check on key for safety
        private const int CursorLgSize = 6;                 // 2^6 = 64 slots (≈512B)
        private const int CursorSize = 1 << CursorLgSize; // 64
        private const int CursorMask = CursorSize - 1;

        // Tune to your allocation grain (64 for 64-multiple instance counts, etc.)
        private const int BucketShift = 5;                 // bucket = offset >> 6

        [StructLayout(LayoutKind.Sequential)]
        private struct CursorCell
        {
            public uint key; // hashed key = Mix32(offset)
            public int prev;  // candidate 'prev' node for that bucket
        }

        private CursorCell* cursor; // unmanaged tiny table; no GC

        public SmallBlockAllocator(int maxInstances)
        {
            this.maxInstances = maxInstances;

            // Node pool with growth
            blockPool = new FreeBlockAllocator(128);

            // Create initial free range covering whole space
            int root = blockPool.AllocateNode();
            var nodes = blockPool.GetUnsafePtr();
            nodes[root].offset = 0;
            nodes[root].count = maxInstances;
            nodes[root].nextID = InvalidID;
            nodes[root].prevID = InvalidID;

            freeListHeadID = root;
            allocCursorID = freeListHeadID;

            totalFree = maxInstances;
            maxFreeBlock = maxInstances;
            maxDirty = false;

            // Init tiny cursor cache
            long csz = (long)UnsafeUtility.SizeOf<CursorCell>() * CursorSize;
            cursor = (CursorCell*)UnsafeUtility.Malloc(csz, UnsafeUtility.AlignOf<CursorCell>(), Allocator.Persistent);
            UnsafeUtility.MemClear(cursor, csz);
            for (int i = 0; i < CursorSize; ++i)
            {
                cursor[i].key = 0xFFFFFFFFu;// invalid key
                cursor[i].prev = InvalidID;
            }
        }

        public void Dispose()
        {
            if (cursor != null)
            {
                UnsafeUtility.Free(cursor, Allocator.Persistent);
                cursor = null;
            }
            blockPool.Dispose();
        }
        
        /// <summary>
        /// Check if allocator is initialized
        /// </summary>
        public bool IsCreated => blockPool.m_Blocks != null;

        /// <summary>
        /// Attempts to allocate a contiguous range of <paramref name="count"/> instances from the free list,
        /// using a next-fit strategy with a two-pass wrap-around scan (cursor→tail, then head→cursor).
        /// </summary>
        /// <returns>Start offset on success; -1 on failure.</returns>
        public int Allocate(int count)
        {
            // Early validation - reject invalid requests immediately
            if (count <= 0) return -1;

            // Quick lower bound check - O(1) early exit for impossible requests
            if (totalFree < count)
                return -1;

            // Failure cache: if structure unchanged and max < request -> O(1) fail
            // This prevents expensive full-list scans for oversized requests
            if (!maxDirty && maxFreeBlock < count)
                return -1;

            var nodes = blockPool.GetUnsafePtr();
            if (freeListHeadID == InvalidID)
                return -1;

            int observedMax = 0; // track largest block seen during scan for failure cache

            // ---- Pass 1: scan from cursor to tail (next-fit for locality) ----
            int cur = allocCursorID;
            while (cur != InvalidID)
            {
                if (nodes[cur].count > observedMax) observedMax = nodes[cur].count;

                if (nodes[cur].count >= count)
                {
                    // Carve from the front of the block
                    int start = nodes[cur].offset;
                    nodes[cur].offset += count;
                    nodes[cur].count -= count;

                    totalFree -= count;
                    maxDirty = true; // structure mutated → invalidate failure cache

                    if (nodes[cur].count == 0)
                    {
                        // Remove empty node in O(1) using doubly-linked pointers
                        int prev = nodes[cur].prevID;
                        int next = nodes[cur].nextID;

                        if (prev == InvalidID) freeListHeadID = next;
                        else nodes[prev].nextID = next;
                        if (next != InvalidID) nodes[next].prevID = prev;

                        blockPool.FreeNode(cur);
                        allocCursorID = next; // next-fit: advance cursor for locality
                    }
                    else
                    {
                        allocCursorID = cur; // next-fit: keep cursor for continued locality
                        if (nodes[cur].count > maxFreeBlock)
                            maxFreeBlock = nodes[cur].count; // optimistic raise (harmless if later invalidated)
                    }
                    return start;
                }

                cur = nodes[cur].nextID;
            }

            // ---- Pass 2: wrap-around scan from head to original cursor ----
            cur = freeListHeadID;
            while (cur != InvalidID && cur != allocCursorID)
            {
                // Track largest block seen to update failure cache on scan completion
                if (nodes[cur].count > observedMax) observedMax = nodes[cur].count;

                // Found suitable block - carve from front to maintain sorted order
                if (nodes[cur].count >= count)
                {
                    // Carve from front to maintain ascending offset order
                    int start = nodes[cur].offset;
                    nodes[cur].offset += count;
                    nodes[cur].count -= count;

                    // Update bookkeeping and invalidate failure cache
                    totalFree -= count;
                    maxDirty = true;

                    if (nodes[cur].count == 0)
                    {
                        // Node fully consumed - unlink in O(1) via doubly-linked list
                        int prev = nodes[cur].prevID;
                        int next = nodes[cur].nextID;

                        if (prev == InvalidID) freeListHeadID = next; // removing head
                        else nodes[prev].nextID = next;  // bypass current node
                        if (next != InvalidID) nodes[next].prevID = prev; // fix back-link

                        // Return node to pool and advance cursor for next-fit locality
                        blockPool.FreeNode(cur);
                        allocCursorID = next;
                    }
                    else
                    {
                        // Keep cursor on shrunken node to exploit spatial locality
                        allocCursorID = cur;

                        // Optimistic cache update - will be corrected if wrong
                        if (nodes[cur].count > maxFreeBlock)
                            maxFreeBlock = nodes[cur].count;
                    }
                    return start;
                }

                // Continue scan until we reach the original cursor position
                cur = nodes[cur].nextID;
            }

            // Full scan failed: cache largest block seen to avoid future scans
            maxFreeBlock = observedMax;
            maxDirty = false; // mark cache as valid
            return -1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void FixCursorOnRemove(int removedId, int fallbackId)
        {
            if (allocCursorID == removedId)
                allocCursorID = (fallbackId != InvalidID) ? fallbackId : freeListHeadID;
        }

        // --------------- tiny 32-bit mixer (avalanche) ---------------
        // Similar to Murmur3 fmix32; cheap and distributes low/high bits.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Mix32(uint x)
        {
            x ^= x >> 16;
            x *= 0x85EBCA6Bu;
            x ^= x >> 13;
            x *= 0xC2B2AE35u;
            x ^= x >> 16;
            return x;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int IndexOf(uint key) => (int)(key & CursorMask);

        /// <summary>
        /// Free a previously allocated contiguous range [offset, offset+count).
        /// Adjacency-first (prefer merging) and a tiny direct-mapped cache to seed the search.
        /// </summary>
        public void Deallocate(int offset, int count)
        {
            // --- Bounds / argument checks ---
            if (count <= 0 || offset < 0 || offset > maxInstances || offset + count > maxInstances)
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Debug.LogError($"SmallBlockAllocator.Deallocate invalid range [{offset},{offset + count}) (max={maxInstances}).");
#endif
                return;
            }

            var nodes = blockPool.GetUnsafePtr();
            bool wasEmpty = (freeListHeadID == InvalidID);

            // --- Locate insertion point prev/cur: try tiny cache first; else normal search ---
            int prev = InvalidID;
            int cur = InvalidID;

            // 1) Tiny cursor cache (index = Mix32(offset) & mask)
            uint key = Mix32((uint)offset);
            int idx = IndexOf(key);

            if (cursor != null && cursor[idx].key == key)
            {
                prev = cursor[idx].prev;

                // Single robust guard: if cached 'prev' no longer satisfies prev.offset < offset,
                // fall back to normal head-based search to avoid missing earlier nodes.
                if (prev != InvalidID && nodes[prev].offset >= offset)
                {
                    prev = InvalidID;
                    cur = freeListHeadID;
                }
                else
                {
                    cur = (prev == InvalidID) ? freeListHeadID : nodes[prev].nextID;
                }

                while (cur != InvalidID && nodes[cur].offset < offset)
                {
                    prev = cur;
                    cur = nodes[cur].nextID;
                }
            }
            else
            {
                // Cache miss → normal search from head
                cur = freeListHeadID;
                while (cur != InvalidID && nodes[cur].offset < offset)
                {
                    prev = cur;
                    cur = nodes[cur].nextID;
                }
            }

            // --- Overlap checks (reject double/partial frees to protect structure) ---
            if (prev != InvalidID && nodes[prev].offset + nodes[prev].count > offset)
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Debug.LogError("SmallBlockAllocator.Deallocate: overlap with previous block (double free or split-free).");
#endif
                return;
            }
            if (cur != InvalidID && offset + count > nodes[cur].offset)
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Debug.LogError("SmallBlockAllocator.Deallocate: overlap with next block (double free or split-free).");
#endif
                return;
            }

            // --- Adjacency-first: merge without allocating a node when possible ---
            bool prevAdj = (prev != InvalidID) && (nodes[prev].offset + nodes[prev].count == offset);
            bool nextAdj = (cur != InvalidID) && (offset + count == nodes[cur].offset);

            // 1) Merge both sides: prev + [offset,count) + cur
            if (prevAdj && nextAdj)
            {
                nodes[prev].count += count + nodes[cur].count;

                int next = nodes[cur].nextID;       // unlink cur
                nodes[prev].nextID = next;
                if (next != InvalidID) nodes[next].prevID = prev;

                blockPool.FreeNode(cur);
                FixCursorOnRemove(/*removed*/ cur, /*fallback*/ prev);

                totalFree += count;
                maxDirty = true;
                if (nodes[prev].count > maxFreeBlock) maxFreeBlock = nodes[prev].count;

                // Update tiny cache (bucket keyed by start of freed range)
                if (cursor != null)
                {
                    cursor[idx].key = key;
                    cursor[idx].prev = prev;
                }

                if (wasEmpty) allocCursorID = freeListHeadID;
                return;
            }

            // 2) Only left-adjacent: extend prev
            if (prevAdj)
            {
                nodes[prev].count += count;

                totalFree += count;
                maxDirty = true;
                if (nodes[prev].count > maxFreeBlock) maxFreeBlock = nodes[prev].count;

                if (cursor != null)
                {
                    cursor[idx].key = key;
                    cursor[idx].prev = prev;
                }

                if (wasEmpty) allocCursorID = freeListHeadID;
                return;
            }

            // 3) Only right-adjacent: grow cur to the left
            if (nextAdj)
            {
                nodes[cur].offset = offset;
                nodes[cur].count += count;

                totalFree += count;
                maxDirty = true;
                if (nodes[cur].count > maxFreeBlock) maxFreeBlock = nodes[cur].count;

                if (cursor != null)
                {
                    // Use cur.prevID as the bucket's candidate 'prev' (most accurate)
                    cursor[idx].key = key;
                    cursor[idx].prev = nodes[cur].prevID;
                }

                if (wasEmpty) allocCursorID = freeListHeadID;
                return;
            }

            // 4) No adjacency: insert a new node
            if (!blockPool.EnsureFreeNode())
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Debug.LogError("SmallBlockAllocator: EnsureFreeNode failed (node pool growth).");
#endif
                return;
            }
            nodes = blockPool.GetUnsafePtr(); // after growth, reload pointer

            int nid = blockPool.AllocateNode();
            if (nid == InvalidID)
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Debug.LogError("SmallBlockAllocator: AllocateNode failed after EnsureFreeNode.");
#endif
                return;
            }

            ref var nb = ref nodes[nid];
            nb.offset = offset;
            nb.count = count;
            nb.prevID = prev;
            nb.nextID = cur;

            if (prev == InvalidID) freeListHeadID = nid;
            else nodes[prev].nextID = nid;
            if (cur != InvalidID) nodes[cur].prevID = nid;

            if (wasEmpty) allocCursorID = freeListHeadID;

            totalFree += count;
            maxDirty = true;
            if (nb.count > maxFreeBlock) maxFreeBlock = nb.count;

            if (cursor != null)
            {
                cursor[idx].key = key;
                cursor[idx].prev = prev; // left neighbor is the canonical 'prev'
            }
        }
    }
}

