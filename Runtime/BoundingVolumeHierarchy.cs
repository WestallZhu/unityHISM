
using Unity.Burst;
using Unity.Mathematics;

using Unity.Collections;

using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;

namespace Renderloom.Rendering
{
    public static class AABBExtensions
    {
        public static float SurfaceArea(this AABB aabb)
        {
            var size = aabb.Extents * 2f;
            return 2.0f * (size.x * size.y + size.y * size.z + size.z * size.x);
        }
    }

    public struct BVHNode
    {
        public AABB BoundingBox;
        public int LeftChildIndex;
        public int RightChildIndex;
        public int FirstInstance;
        public int LastInstance;
    }

    public struct Primitive
    {
        public AABB BoundingBox;
        public int index;
    }
    public unsafe struct BoundingVolumeHierarchy
    {

        public void BuildBVH(NativeArray<BVHNode> nodes, NativeArray<Primitive> primitives, int nodeIndex, int start, int end, int depth)
        {
            BVHNode* node = (BVHNode*)nodes.GetUnsafePtr() + nodeIndex;

            node->BoundingBox = ComputeBoundingBox(primitives, start, end);

            int primitiveCount = end - start;

            if (primitiveCount <= 4 || depth >= 4)
            {
                node->LeftChildIndex = -1;
                node->RightChildIndex = -1;
                node->FirstInstance = start;
                node->LastInstance = end;
                return;
            }

            int mid = PartitionPrimitives(primitives, start, end);

            int leftChildIndex = nodeIndex + 1;
            int rightChildIndex = nodeIndex + 2 * (mid - start);

            node->LeftChildIndex = leftChildIndex;
            node->RightChildIndex = rightChildIndex;

            BuildBVH(nodes, primitives, leftChildIndex, start, mid, depth + 1);
            BuildBVH(nodes, primitives, rightChildIndex, mid, end, depth + 1);

            node->FirstInstance = nodes[leftChildIndex].FirstInstance;
            node->LastInstance = nodes[rightChildIndex].LastInstance;
        }

        public int PartitionPrimitives(NativeArray<Primitive> primitives, int start, int end)
        {
            AABB combinedBox = ComputeBoundingBox(primitives, start, end);
            float totalArea = combinedBox.SurfaceArea();

            int bestSplitIndex = start;
            float minCost = float.MaxValue;

            for (int axis = 0; axis < 3; axis++)
            {
                var slice = primitives.Slice(start, end - start);
                slice.Sort(new PrimitiveCentroidComparer(axis));

                for (int i = start + 1; i < end; i++)
                {
                    AABB leftBox = ComputeBoundingBox(primitives, start, i);
                    AABB rightBox = ComputeBoundingBox(primitives, i, end);

                    float leftArea = leftBox.SurfaceArea();
                    float rightArea = rightBox.SurfaceArea();

                    // Calculate cost (SAH)
                    float cost = (i - start) * leftArea + (end - i) * rightArea;

                    if (cost < minCost)
                    {
                        minCost = cost;
                        bestSplitIndex = i;
                    }
                }
            }

            return bestSplitIndex;
        }

        private struct PrimitiveCentroidComparer : IComparer<Primitive>
        {
            private readonly int axis;

            public PrimitiveCentroidComparer(int axis)
            {
                this.axis = axis;
            }

            public int Compare(Primitive x, Primitive y)
            {
                float centroidX = x.BoundingBox.Center[axis];
                float centroidY = y.BoundingBox.Center[axis];
                return centroidX.CompareTo(centroidY);
            }
        }

        public AABB ComputeBoundingBox(NativeArray<Primitive> primitives, int start, int end)
        {
            var bounds = MinMaxAABB.Empty;
            for (int i = start; i < end; i++)
            {
                bounds.Encapsulate(primitives[i].BoundingBox);
            }
            return bounds;
        }

    }
}
