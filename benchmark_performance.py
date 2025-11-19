#!/usr/bin/env python3
"""
Performance Benchmarking Script for Analogico

This script benchmarks the performance of the Analogico project with and without
the performance optimizations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import time
import numpy as np
from src.hp_inv import hp_inv
from src.rram_model import create_rram_matrix
from src.performance_optimization import (
    PerformanceOptimizer, 
    MixedPrecisionOptimizer,
    SparseRRAMModel,
    jax_hp_inv,
    parallel_stress_test
)
from src.stress_test import run_stress_test

def benchmark_hp_inv_performance():
    """Benchmark HP-INV performance with different optimizations."""
    print("HP-INV Performance Benchmarking")
    print("=" * 40)
    
    # Test different matrix sizes
    sizes = [32, 64, 128]
    
    for size in sizes:
        print(f"\nBenchmarking {size}x{size} matrices:")
        
        # Create test matrices
        G = create_rram_matrix(size, variability=0.03, stuck_fault_prob=0.01)
        G = G + 0.1 * np.eye(size)  # Ensure conditioning
        b = np.random.randn(size)
        
        # Benchmark standard implementation
        start_time = time.time()
        x_std, iters_std, info_std = hp_inv(G, b, max_iter=10)
        std_time = time.time() - start_time
        print(f"  Standard HP-INV: {std_time:.4f}s, {iters_std} iterations")
        
        # Test mixed precision optimization
        mixed_opt = MixedPrecisionOptimizer()
        start_time = time.time()
        x_mixed, iters_mixed, info_mixed = mixed_opt.hp_inv_mixed_precision(G, b, max_iter=10)
        mixed_time = time.time() - start_time
        print(f"  Mixed Precision: {mixed_time:.4f}s, {iters_mixed} iterations")
        
        # Test JAX implementation if available
        if 'jax' in sys.modules:
            try:
                start_time = time.time()
                x_jax, iters_jax, info_jax = jax_hp_inv(G, b, max_iter=10)
                jax_time = time.time() - start_time
                print(f"  JAX GPU/CPU: {jax_time:.4f}s, {iters_jax} iterations")
            except:
                print(f"  JAX optimization: Not available")
        else:
            print(f"  JAX optimization: Not available")
            
        # Calculate speedups
        if mixed_time > 0:
            mixed_speedup = std_time / mixed_time if mixed_time > 0 else 0
            print(f"  Mixed Precision Speedup: {mixed_speedup:.2f}x")


def benchmark_stress_testing():
    """Benchmark stress testing performance."""
    print("\n\nStress Testing Performance Benchmarking")
    print("=" * 50)
    
    # Parameters for stress testing
    n = 8
    num_samples = 50
    
    print(f"Running stress tests with {num_samples} samples on {n}x{n} matrices:")
    
    # Sequential stress test
    start_time = time.time()
    iters_seq, errors_seq = run_stress_test(n=n, num_samples=num_samples, use_parallel=False)
    seq_time = time.time() - start_time
    print(f"  Sequential: {seq_time:.4f}s for {num_samples} samples ({len(iters_seq)} successful)")
    
    # Parallel stress test with multiprocessing
    start_time = time.time()
    iters_par, errors_par = run_stress_test(n=n, num_samples=num_samples, use_parallel=True, n_processes=4)
    par_time = time.time() - start_time
    print(f"  Parallel (4 cores): {par_time:.4f}s for {num_samples} samples ({len(iters_par)} successful)")
    
    if par_time > 0:
        speedup = seq_time / par_time if par_time > 0 else 0
        print(f"  Parallel Speedup: {speedup:.2f}x")


def benchmark_sparse_operations():
    """Benchmark sparse matrix operations when available."""
    print("\n\nSparse Matrix Performance")
    print("=" * 30)
    
    sparse_model = SparseRRAMModel()
    if sparse_model.sparse_available:
        import scipy.sparse as sp
        
        # Create sparse matrix
        n = 256
        sparsity = 0.05  # 5% non-zero elements
        G_sparse = sparse_model.create_sparse_rram_matrix(n, sparsity=sparsity)
        
        if G_sparse is not None:
            b = np.random.randn(n)
            
            # Dense matrix operation
            start_time = time.time()
            x_dense = np.linalg.solve(G_sparse, b)
            dense_time = time.time() - start_time
            print(f"  Dense operation ({n}x{n}, {sparsity*100:.1f}% sparse): {dense_time:.4f}s")
            
            # The sparse optimization would use scipy.sparse matrices directly
            print(f"  Sparse optimization: Available but not implemented in this benchmark")
        else:
            print("  Sparse matrices: Not available (scipy not installed)")
    else:
        print("  Sparse matrices: Not available (scipy not installed)")


def benchmark_optimization_tools():
    """Benchmark the optimization tools.""" 
    print("\n\nOptimization Tools Benchmarking")
    print("=" * 40)
    
    optimizer = PerformanceOptimizer()
    
    print(f"  JAX available: {optimizer.use_jax}")
    print(f"  Numba available: {optimizer.use_numba}")
    print(f"  Using {optimizer.n_cores} CPU cores")
    
    # Test quantization optimization
    matrix = np.random.rand(100, 100)
    
    # Standard quantization
    start_time = time.time()
    for _ in range(100):
        max_val = np.max(np.abs(matrix))
        if max_val == 0:
            result = matrix
        else:
            levels = 2**3 - 1
            scale = levels / max_val
            result = np.round(matrix * scale) / scale
    std_time = time.time() - start_time
    print(f"  Standard quantization (100x): {std_time:.4f}s")
    
    # Optimized quantization (if available)
    quantize_func, _ = optimizer.optimize_rram_model()
    
    start_time = time.time()
    for _ in range(100):
        _ = quantize_func(matrix, 3)
    opt_time = time.time() - start_time
    print(f"  Optimized quantization (100x): {opt_time:.4f}s")
    
    if opt_time > 0:
        speedup = std_time / opt_time if opt_time > 0 else 0
        print(f"  Quantization speedup: {speedup:.2f}x")


def main():
    """Run all benchmarks."""
    print("Analogico Performance Optimization Benchmarking Suite")
    print("=" * 60)
    
    # Run all benchmarks
    benchmark_optimization_tools()
    benchmark_hp_inv_performance()
    benchmark_stress_testing()
    benchmark_sparse_operations()
    
    print("\n" + "=" * 60)
    print("Benchmarking complete!")
    print("\nKey Performance Improvements Implemented:")
    print("- Parallel processing for stress tests using multiprocessing")
    print("- JAX-based GPU/CPU acceleration for HP-INV")
    print("- Mixed precision computing for performance/accuracy trade-offs")
    print("- Numba JIT compilation for CPU optimization")
    print("- Sparse matrix support for large systems")
    print("- Optimized matrix operations")


if __name__ == "__main__":
    main()