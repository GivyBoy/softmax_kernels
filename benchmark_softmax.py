import torch
import numpy as np
import time
import argparse
import os
import sys

# Import the PyTorch benchmark function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pytorch_softmax_benchmark import benchmark_softmax

# Build and import the CUDA extension
try:
    import softmax_cuda
except ImportError:
    print("Building CUDA extension...")
    setup_result = os.system("python setup.py build_ext --inplace")
    if setup_result != 0:
        raise RuntimeError("Failed to build CUDA extension")
    import softmax_cuda

import softmax_cuda


def benchmark_custom_softmax(implementation, input_tensor, num_runs=100, warmup_runs=10):
    """
    Benchmark a custom softmax implementation.

    Args:
        implementation: Function to benchmark
        input_tensor: Input tensor
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    """
    implementation_name = implementation.__name__.replace("_", " ").title()
    rows, cols = input_tensor.shape
    print(f"Benchmarking {implementation_name} with matrix size: {rows}x{cols}")

    # Warmup runs
    for _ in range(warmup_runs):
        _ = implementation(input_tensor)

    # Make sure warmup is complete
    torch.cuda.synchronize()

    times = []

    for _ in range(num_runs):
        torch.cuda.synchronize()

        start = time.perf_counter()
        output = implementation(input_tensor)

        torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    times = np.array(times)

    print(f"Results (ms):")
    print(f"  Mean time:   {np.mean(times):.4f}")
    print(f"  Min time:    {np.min(times):.4f}")
    print(f"  Std dev:     {np.std(times):.4f}")

    return output, times


def check_correctness(pytorch_output, custom_output, rtol=1e-5, atol=1e-6):
    """
    Check if two tensors are approximately equal.

    Args:
        pytorch_output: PyTorch output tensor
        custom_output: Custom implementation output tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    if torch.allclose(pytorch_output, custom_output, rtol=rtol, atol=atol):
        print("Outputs match within tolerance!")
    else:
        max_diff = torch.max(torch.abs(pytorch_output - custom_output))
        mean_diff = torch.mean(torch.abs(pytorch_output - custom_output))
        print(f"Outputs differ! Max difference: {max_diff:.6e}, Mean difference: {mean_diff:.6e}")

        # Print some sample values for debugging
        sample_idx = torch.argmax(torch.abs(pytorch_output - custom_output)).item()
        row, col = sample_idx // pytorch_output.shape[1], sample_idx % pytorch_output.shape[1]
        print(f"Sample difference at [{row}, {col}]:")
        print(f"  PyTorch:  {pytorch_output[row, col].item():.10f}")
        print(f"  Custom:   {custom_output[row, col].item():.10f}")

        # Check if outputs sum to 1 along dimension 1
        pytorch_sum = torch.sum(pytorch_output, dim=1)
        custom_sum = torch.sum(custom_output, dim=1)
        print(f"PyTorch sum range: [{pytorch_sum.min().item():.10f}, {pytorch_sum.max().item():.10f}]")
        print(f"Custom sum range:  [{custom_sum.min().item():.10f}, {custom_sum.max().item():.10f}]")


def run_benchmarks(rows, cols, num_runs=10, warmup_runs=5):
    """
    Run benchmarks for all implementations.

    Args:
        rows: Number of rows in the input matrix
        cols: Number of columns in the input matrix
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    """
    print(f"Creating random input tensor of size {rows}x{cols}")
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda")

    # PyTorch benchmark
    print("\n=== PyTorch Softmax Benchmark ===")
    pytorch_times = benchmark_softmax(rows, cols, num_runs, warmup_runs, device="cuda")
    pytorch_output = torch.softmax(x, dim=1)

    # Custom implementation benchmarks
    print("\n=== Naive Softmax Benchmark ===")
    naive_output, naive_times = benchmark_custom_softmax(softmax_cuda.naive_softmax, x, num_runs, warmup_runs)

    print("\n=== Online Softmax Benchmark ===")
    online_output, online_times = benchmark_custom_softmax(softmax_cuda.online_softmax, x, num_runs, warmup_runs)

    print("\n=== Shared Memory Softmax Benchmark ===")
    shared_mem_output, shared_mem_times = benchmark_custom_softmax(
        softmax_cuda.online_softmax_shared_mem, x, num_runs, warmup_runs
    )

    print("\n=== Shared Memory (w/ vectorization) Softmax Benchmark ===")
    shared_mem_vectorized_output, shared_mem_vectorized_times = benchmark_custom_softmax(
        softmax_cuda.online_softmax_shared_mem_vectorized, x, num_runs, warmup_runs
    )

    print("\n=== Manual Vectorized Softmax Benchmark ===")
    vectorized_output, vectorized_times = benchmark_custom_softmax(
        softmax_cuda.online_softmax_vectorized, x, num_runs, warmup_runs
    )

    print("\n=== Warp Optimized Softmax Benchmark ===")
    warp_optimized_output, warp_optimized_times = benchmark_custom_softmax(
        softmax_cuda.online_softmax_warp_optimized, x, num_runs, warmup_runs
    )

    print("\n=== Cooperative Groups Softmax Benchmark ===")
    cooperative_output, cooperative_times = benchmark_custom_softmax(
        softmax_cuda.online_softmax_cooperative, x, num_runs, warmup_runs
    )

    # Check correctness
    print("\n=== Correctness Check ===")
    print("Comparing PyTorch vs Naive Softmax:")
    check_correctness(pytorch_output, naive_output)

    print("\nComparing PyTorch vs Online Softmax:")
    check_correctness(pytorch_output, online_output)

    print("\nComparing PyTorch vs Shared Memory Softmax:")
    check_correctness(pytorch_output, shared_mem_output)

    print("\nComparing PyTorch vs Shared Memory (w/ vectorization) Softmax:")
    check_correctness(pytorch_output, shared_mem_vectorized_output)

    print("\nComparing PyTorch vs Manual Vectorized Softmax:")
    check_correctness(pytorch_output, vectorized_output)

    print("\nComparing PyTorch vs Warp Optimized Softmax:")
    check_correctness(pytorch_output, warp_optimized_output)

    print("\nComparing PyTorch vs Cooperative Groups Softmax:")
    check_correctness(pytorch_output, cooperative_output)

    # Performance comparison
    print("\n=== Performance Comparison ===")
    print(f"{'Implementation':<25} {'Mean Time (ms)':<15} {'Min Time (ms)':<15} {'Speedup vs PyTorch':<20}")
    print("-" * 75)

    pytorch_mean = np.mean(pytorch_times)
    pytorch_min = np.min(pytorch_times)

    print(f"{'PyTorch':<25} {pytorch_mean:.4f}{'ms':<10} {pytorch_min:.4f}{'ms':<10} {'1.00x':<20}")

    naive_mean = np.mean(naive_times)
    naive_min = np.min(naive_times)
    print(
        f"{'Naive Softmax':<25} {naive_mean:.4f}{'ms':<10} {naive_min:.4f}{'ms':<10} {pytorch_mean/naive_mean:.2f}{'x':<20}"
    )

    online_mean = np.mean(online_times)
    online_min = np.min(online_times)
    print(
        f"{'Online Softmax':<25} {online_mean:.4f}{'ms':<10} {online_min:.4f}{'ms':<10} {pytorch_mean/online_mean:.2f}{'x':<20}"
    )

    shared_mean = np.mean(shared_mem_times)
    shared_min = np.min(shared_mem_times)
    print(
        f"{'Shared Memory Softmax':<25} {shared_mean:.4f}{'ms':<10} {shared_min:.4f}{'ms':<10} {pytorch_mean/shared_mean:.2f}{'x':<20}"
    )

    shared_mem_vectorized_mean = np.mean(shared_mem_vectorized_times)
    shared_mem_vectorized_min = np.min(shared_mem_vectorized_times)
    print(
        f"{'Shared Memory (w/ vectorization) Softmax':<25} {shared_mem_vectorized_mean:.4f}{'ms':<10} {shared_mem_vectorized_min:.4f}{'ms':<10} {pytorch_mean/shared_mem_vectorized_mean:.2f}{'x':<20}"
    )

    vectorized_mean = np.mean(vectorized_times)
    vectorized_min = np.min(vectorized_times)
    print(
        f"{'Manual Vectorized Softmax':<35} {vectorized_mean:.4f}{'ms':<10} {vectorized_min:.4f}{'ms':<10} {pytorch_mean/vectorized_mean:.2f}{'x':<20}"
    )

    warp_optimized_mean = np.mean(warp_optimized_times)
    warp_optimized_min = np.min(warp_optimized_times)
    print(
        f"{'Warp Optimized Softmax':<35} {warp_optimized_mean:.4f}{'ms':<10} {warp_optimized_min:.4f}{'ms':<10} {pytorch_mean/warp_optimized_mean:.2f}{'x':<20}"
    )

    cooperative_mean = np.mean(cooperative_times)
    cooperative_min = np.min(cooperative_times)
    print(
        f"{'Cooperative Groups Softmax':<35} {cooperative_mean:.4f}{'ms':<10} {cooperative_min:.4f}{'ms':<10} {pytorch_mean/cooperative_mean:.2f}{'x':<20}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark softmax implementations")
    parser.add_argument("--rows", type=int, default=4096, help="Number of rows in the input matrix")
    parser.add_argument("--cols", type=int, default=4096, help="Number of columns in the input matrix")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")

    args = parser.parse_args()

    run_benchmarks(args.rows, args.cols, args.runs, args.warmup)
