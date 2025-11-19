"""
Performance Optimization Examples for Analogico

This file demonstrates how to use the performance optimization features in the Analogico project.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import time
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


def example_1_mixed_precision():
    """Example 1: Using mixed precision for performance."""
    print("=== Example 1: Mixed Precision Optimization ===")
    
    # Create a larger matrix to see performance benefits
    n = 128
    G = create_rram_matrix(n, variability=0.03, stuck_fault_prob=0.01)
    G = G + 0.1 * np.eye(n)  # Ensure conditioning
    b = np.random.randn(n)
    
    # Standard precision
    start_time = time.time()
    x_std, iters_std, info_std = hp_inv(G, b, max_iter=10)
    std_time = time.time() - start_time
    print(f"Standard precision: {std_time:.4f}s, {iters_std} iterations")
    
    # Mixed precision
    mixed_opt = MixedPrecisionOptimizer()
    start_time = time.time()
    x_mixed, iters_mixed, info_mixed = mixed_opt.hp_inv_mixed_precision(G, b, max_iter=10)
    mixed_time = time.time() - start_time
    print(f"Mixed precision: {mixed_time:.4f}s, {iters_mixed} iterations")
    
    # Compare results
    solution_diff = np.linalg.norm(x_std - x_mixed) / np.linalg.norm(x_std)
    print(f"Solution difference: {solution_diff:.2e}")
    print(f"Speedup: {std_time/mixed_time:.2f}x")
    print()


def example_2_parallel_stress_test():
    """Example 2: Using parallel processing for stress tests."""
    print("=== Example 2: Parallel Stress Testing ===")
    
    # For meaningful parallelism, we need larger/more complex tests
    # In this case, we'll run more samples to see the benefit
    
    n = 8
    num_samples = 100  # More samples to benefit from parallelization
    
    print(f"Running stress test with {num_samples} samples on {n}x{n} matrices...")
    
    # Sequential stress test
    start_time = time.time()
    iters_seq, errors_seq = run_stress_test(
        n=n, 
        num_samples=num_samples, 
        use_parallel=False
    )
    seq_time = time.time() - start_time
    print(f"Sequential: {seq_time:.4f}s for {num_samples} samples ({len(iters_seq)} successful)")
    
    # Parallel stress test
    start_time = time.time()
    iters_par, errors_par = run_stress_test(
        n=n, 
        num_samples=num_samples, 
        use_parallel=True,
        n_processes=4
    )
    par_time = time.time() - start_time
    print(f"Parallel (4 cores): {par_time:.4f}s for {num_samples} samples ({len(iters_par)} successful)")
    
    if par_time > 0:
        speedup = seq_time / par_time if par_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
    print()


def example_3_performance_optimizer():
    """Example 3: Using the PerformanceOptimizer class."""
    print("=== Example 3: Performance Optimizer Class ===")
    
    optimizer = PerformanceOptimizer()
    quantize_func, mvm_func = optimizer.optimize_rram_model()
    
    print(f"JAX available: {optimizer.use_jax}")
    print(f"Numba available: {optimizer.use_numba}")
    print(f"Using {optimizer.n_cores} CPU cores")
    
    # Example usage of optimized functions
    matrix = np.random.rand(50, 50)
    vector = np.random.rand(50)
    
    # Standard operations
    start_time = time.time()
    for _ in range(100):
        _ = np.round(matrix * (7 / np.max(np.abs(matrix)))) * np.max(np.abs(matrix)) / 7
    std_time = time.time() - start_time
    
    print(f"Standard operations timing: {std_time:.4f}s for 100 operations")
    print()


def example_4_optimized_workflow():
    """Example 4: Complete optimized workflow."""
    print("=== Example 4: Complete Optimized Workflow ===")
    
    # Create multiple RRAM matrices with different conditions
    matrices = []
    for i in range(5):
        G = create_rram_matrix(32, variability=0.03, stuck_fault_prob=0.01)
        G = G + 0.1 * np.eye(32)  # Ensure conditioning
        matrices.append(G)
    
    b = np.random.randn(32)
    
    # Solve all systems using mixed precision for efficiency
    mixed_opt = MixedPrecisionOptimizer()
    results = []
    
    start_time = time.time()
    for i, G in enumerate(matrices):
        x, iters, info = mixed_opt.hp_inv_mixed_precision(G, b, max_iter=10)
        results.append((x, iters, info))
        print(f"  Solved system {i+1}: {iters} iterations")
    
    total_time = time.time() - start_time
    print(f"Total time for 5 systems: {total_time:.4f}s")
    
    # Average iterations
    avg_iters = sum(r[1] for r in results) / len(results)
    print(f"Average iterations: {avg_iters:.1f}")
    print()


def example_5_large_scale_simulation():
    """Example 5: Large-scale simulation with optimizations."""
    print("=== Example 5: Large-Scale Simulation ===")
    
    print("Performing large-scale analysis with parallel processing...")
    
    # Parameters for large-scale test
    n = 16  # Larger matrix than before
    num_samples = 50  # More samples
    
    print(f"Running stress test: {num_samples} samples of {n}x{n} matrices")
    
    # Parallel execution with more samples to see benefits
    start_time = time.time()
    iters, errors = run_stress_test(
        n=n,
        num_samples=num_samples,
        use_parallel=True,
        n_processes=4,
        variability=0.05,
        stuck_prob=0.02
    )
    par_time = time.time() - start_time
    
    print(f"Completed in: {par_time:.4f}s")
    print(f"Successful runs: {len(iters)}/{num_samples}")
    print(f"Average iterations: {np.mean(iters):.2f}")
    print(f"Average error: {np.mean(errors):.2e}")
    print()


def main():
    """Run all performance optimization examples."""
    print("Analogico Performance Optimization Examples")
    print("=" * 50)
    
    example_1_mixed_precision()
    example_2_parallel_stress_test()
    example_3_performance_optimizer()
    example_4_optimized_workflow()
    example_5_large_scale_simulation()
    
    print("=" * 50)
    print("Performance optimization examples completed!")
    print("\nKey Capabilities Demonstrated:")
    print("- Mixed precision computing for speed/accuracy trade-offs")
    print("- Parallel processing for large-scale simulations")
    print("- Performance optimization tools and utilities")
    print("- Optimized workflows for better efficiency")


if __name__ == "__main__":
    main()