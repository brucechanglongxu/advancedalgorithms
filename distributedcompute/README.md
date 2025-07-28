1. https://sumanthrh.com/post/distributed-and-efficient-finetuning/#:~:text=2,is%20placed%20on%203%20GPUs
2. https://huggingface.co/docs/transformers/perf_train_gpu_many
3. https://lilianweng.github.io/posts/2021-09-25-train-large/

## Data-Parallelism

The most naive way for Data parallelism is to copy the model weight sinto multiple workers and assign a fraction of data to each worker to be processed at the same time. Naive DP cannot work well if the model size is larger than a single GPU node’s memory. Methods like GeePS (Cui et al. 2016) offload temporarily unused parameters back to CPU to work with limited GPU memory when the model is too big to fit into one machine. The data swapping transfer should happen at the backend and not interfere with training computation. At the end of each minibatch, workers need to synchronize gradients or weights to avoid staleness. There are two main synchronization approaches and both have clear pros & cons.

1. **Bulk synchronous parallels (BSP):** Workers sync data at the end of every minibatch. It prevents model weights staleness and good learning efficiency but each machine has to halt and wait for others to send gradients.
2. **Asynchronous parallel (ASP):** Every GPU worker processes the data asynchronously, no waiting or stalling. However, it can easily lead to stale weights being used and thus lower the statistical learning efficiency. Even though it increases the computation time, it may not speed up training time to convergence.

<img width="1738" height="1936" alt="image" src="https://github.com/user-attachments/assets/89a47bba-94de-47a1-9cd2-8cdb64be9176" />

Somewhere in the middle we can synchronize gradients globally once every $$x$$ iterations ($$x > 1$$). This is called _"gradient accumulation"_ in Distributed Data Parallel (DDP) since Pytorch v1.5 (Li et al. 2021). Bucketing gradients avoid immediate `AllReduce` operations but instead buckets multiple gradients into one `AllReduce` to improve throughput. Computation and communication scheduling optimization can be made based on the computation graph. 

<img width="951" height="392" alt="image" src="https://github.com/user-attachments/assets/79661557-5ad3-437c-aeb8-5c8a04dfad78" />

1. **Stage 1** partitions the optimizer states
2. **Stage 2** partitions the optimizer and gradient states
3. **Stage 3** partitions the optimizer, gradient and parameters

DeepSpeed's [ZeRO](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) (Zero Redundancy Optimizer) is a form of _data parallelism_ that massively improves on memory efficiency. The main idea is that ZeRO exploits memory redundancy in data-parallel training and the latest improvements in fast inter-GPU communication to improve throughput, with some increase in communication volume, depending on the stage. ZeRO has two components - ZeRO-DP (data parallelism) and ZeRO-R (residual memory). The DeepSpeed team 
