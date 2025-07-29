# [NVIDIA GPU Architecture](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

It is helpful to understand the basics of GPU execution when reasoning about how efficiently particular layers or neural networks are utilizing a given GPU. First we will dive into the basic structure of GPU architecture. Ultimately, the GPU is a _highly parallel processor_ that is composed of processing elements and a memory hierarchy. At a high-level there are multiple **Streaming Multiprocessors (SMs)**, an on-chip L2 cache, and high-bandwidth DRAM. 

![Alt text](image.png)