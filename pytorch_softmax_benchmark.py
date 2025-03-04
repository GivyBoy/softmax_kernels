import torch
import numpy as np
import time


def benchmark_softmax(rows, cols, num_runs=100, warmup_runs=10, device="cuda"):
    """
    Simple benchmark for PyTorch's softmax implementation.

    Args:
        rows: Number of rows in the matrix
        cols: Number of columns in the matrix
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: 'cuda' or 'cpu'
    """
    print(f"Benchmarking softmax on {device} with matrix size: {rows}x{cols}")

    # create random input tensor
    x = torch.randn(rows, cols, device=device)

    # warmup runs
    for _ in range(warmup_runs):
        _ = torch.softmax(x, dim=1)

    # make sure warmup is complete
    if device == "cuda":
        torch.cuda.synchronize()

    times = []

    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = torch.softmax(x, dim=1)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # convert to milliseconds

    times = np.array(times)

    print(f"Results (ms):")
    print(f"  Mean time:   {np.mean(times):.4f}")
    print(f"  Min time:    {np.min(times):.4f}")
    print(f"  Std dev:     {np.std(times):.4f}")

    return times


if __name__ == "__main__":
    rows = 4096
    cols = 4096
    runs = 10
    warmup = 1

    benchmark_softmax(rows, cols, runs, warmup)
