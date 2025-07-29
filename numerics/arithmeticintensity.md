# Arithmetic Intensity

_Arithmetic intensity_ is a metric calculated across the entire operation/kernel with a global scope-across the GPU. Each SM gets assigned thread blocks at runtime, and executes some chunk of the total work - contributing to the global FLOPs and memory traffic. The same kernel (i.e. code and data shape) runs acaross all SMs, so each SM executes a representative chunk of the same arithmetic vs memory behavior (i.e. same arithmetic intensity). Therefore, we reason about arithmetic intensity _per-kernel_ and compare it to the GPU's overall math:memory ratio (e.g. 312 TFLOPs / 2 TB/s = 156 FLOPs/byte on A100). To formalize this:

$$\textbf{Arithmetic Intensity} = \frac{\textbf{Total FLOPs performed}}{\textbf{Total bytes transferred to/from memory}}$$

It answers the following question:
```
For every byte I load from memory, how much math do I do before I need another one?
```
it's sort of like asking _"am I thinking hard about every piece of information I learn?"_ or _"am I mindlessly munching data without doing much with it?"_. From a human learning perspective, if we were all processing elements and trying to learn new concept from teacher (i.e. compute kernel launched) - **we want a higher arithmetic intensity** so that we are getting the most understanding/knowledge from each piece of information consumed.[^1] 

Ultimately, a GPU has two main drivers of performance - math throughput (how fast it can do FLOPs) and memory bandwidth (how fast it can move bytes). We want our program to feed the math units faster than memory can become a bottleneck (as human students, we want to learn faster than the knowledge that the syllabus/teacher is presenting to us, to be an A+ student). Hence high arithmetic intensity implies that _we are working hard on fewer bytes_ (good!) whereas low arithmetic intensity implies that _we are starving math units and just moving bytes around without true work_ (bad!). Another analogy is: imaging our chef (GPU core) is cooking dishes (doing math), ingredients (data) come from a pantry (global memory). If the chef needs to run back to the pantry after every chop, the coking slows down. A few principles to keep in mind:

1. High arithmetic intensity can be amplified by _caching_ and _reuse_ i.e. if we reuse the same data across threads, we "pay" the memory cost once but do math many times, boosting arithmetic intensity.
2. If we constantly access new, uncached data, we are limited by _memory bandwidth_, which is what kills intensity in element-wise ops like ReLU or layer norm. 

Think of arithmetic intensity as a knob we can turn. Left is _"memory limited"_ (ReLU, batch norm, low reuse) and right is _"math limited"_ (matmul, attention, big convolutions). The farther right we go, the more likely our GPU's math units (e.g. Tensore Cores) are doing real work. To increase arithmetic intensity we have the following options:

1. **Increase reuse:** Use shared memory, loop tiling or batching (e.g. FlashAttention)
2. **Do more math per byte:** Fuse multiple operations together (e.g. bias/ReLU/matmul or MMA)
3. **Reduce memory footprint:** Use quantization (`int8`), compression, or sparsity
4. **Avoid recomputation or scatter-gather:** These patterns hurt reuse and drive down intensity.

[^1]: It may be worthwhile to distinguish information and knowledge here, though this could lead to an entirely new post about epistemiology. 
