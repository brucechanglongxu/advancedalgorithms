# GPU Ops Demo: Tensor Cores vs CUDA Cores

This minimal example contrasts two GPU execution paths in PyTorch:

- **MatMul** (uses Tensor Cores if FP16 and aligned sizes)
- **ReLU** (element-wise, uses CUDA Cores only)

## Usage

```bash
cd distributedcompute/gpu_ops_demo
pip install -r requirements.txt
python matmul_vs_relu.py

We use `torch.float16` and large aligned tensors (1024x1024) to trigger Tensor Core usage. We can verify kernel execution with `nvprof`, `nsight compute`, or `nsys`. To verify Tensor Core usage:
```bash
nsys profile -t cuda python matmul_vs_relu.py
```

