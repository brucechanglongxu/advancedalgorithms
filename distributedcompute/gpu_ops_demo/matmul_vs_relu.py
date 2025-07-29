import torch
import time

def time_op(label, fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    print(f"{label}: {start.elapsed_time(end):.3f} ms")

def main():
    device = "cuda"
    dtype = torch.float16

    A = torch.randn(1024, 1024, device=device, dtype=dtype)
    B = torch.randn(1024, 1024, device=device, dtype=dtype)
    X = torch.randn(1024, 1024, device=device, dtype=dtype)

    def matmul_op():
        for _ in range(100):
            torch.matmul(A, B)

    def relu_op():
        for _ in range(100):
            torch.relu(X)

    time_op("MatMul (Tensor Core eligible)", matmul_op)
    time_op("ReLU (CUDA Core only)", relu_op)

if __name__ == "__main__":
    main()

