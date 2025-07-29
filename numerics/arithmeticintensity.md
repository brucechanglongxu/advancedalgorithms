# Arithmetic Intensity

_Arithmetic intensity_ is a metric calculated across the entire operation/kernel with a global scope-across the GPU. Each SM gets assigned thread blocks at runtime, and executes some chunk of the total work - contributing to the global FLOPs and memory traffic. The same kernel (i.e. code and data shape) runs acaross all SMs, so each SM executes a representative chunk of the same arithmetic vs memory behavior (i.e. same arithmetic intensity). Therefore, we reason about arithmetic intensity _per-kernel_ and compare it to the GPU's overall math:memory ratio (e.g. 312 TFLOPs / 2 TB/s = 156 FLOPs/byte on A100). To formalize this:

$$\textbf{Arithmetic Intensity} = \frac{\textbf{Total FLOPs performed}}{\textbf{Total bytes transferred to/from memory}}$$

