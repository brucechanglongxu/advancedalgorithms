1. https://sumanthrh.com/post/distributed-and-efficient-finetuning/#:~:text=2,is%20placed%20on%203%20GPUs
2. https://huggingface.co/docs/transformers/perf_train_gpu_many
3. https://lilianweng.github.io/posts/2021-09-25-train-large/

## Data-Parallelism

DeepSpeed's ZeRO (Zero Redundancy Optimizer) is a form of _data parallelism_ that massively improves on memory efficiency. The main idea is that ZeRO exploits memory redundancy in data-parallel training and the latest improvements in fast inter-GPU communication to improve throughput, with some increase in communication volume, depending on the stage. ZeRO has two components - ZeRO-DP (data parallelism) and ZeRO-R (residual memory). The 
