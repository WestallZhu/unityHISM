# Renderloom HISM (Unity SRP)

Hierarchical Instanced Static Mesh rendering built on Unity’s BatchRendererGroup (BRG) for large‑scale static mesh rendering with efficient culling, LOD, and  cbuffer gpu scene batching. Good compatibility for mobile.

## 核心技术点

- 基于 BVH 的视锥裁剪与 LOD 拣选：快速剔除与分级。
- 面向数据的高性能Mesh GPU Resident Drawer: 使用高兼容性的 ConstBuffer GPU Scene + SRPBatch。
- 跨 Chunk 合批优化：64k ContBuffer windows 再分配，结合 FixedSizeAllocator/SmallBlockAllocator/SubBatchAllocator/IntrusiveUnitBucketAllocator 实现内存局部性优化、低抖动的增量更新。
- Chunk 流式分块：按网格单元切片为 Chunk，使用 `AsyncReadManager` 异步加载与零拷贝 Blob 布局。

![image](https://github.com/user-attachments/assets/8e3d2a21-fead-49b1-be28-bce5eadd7276)
