"""
Benchmarking module for Analogico project.

This module provides functionality to benchmark the performance of HP-INV algorithms
against digital implementations and measure various performance metrics.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass

# Import from the project
import sys
import os
# Add the project root directory (parent of src) to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv, blockamc_inversion, recursive_block_inversion
from src.rram_model import create_rram_matrix
from src.redundancy import apply_redundancy
from src.hardware_interface import MockRRAMHardware


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    name: str
    matrix_size: int
    time_elapsed: float
    residual: float
    iterations: int
    error: float
    condition_number: float
    additional_info: Dict = None


class BenchmarkSuite:
    """Suite for running various benchmarks."""
    
    def __init__(self):
        self.results = []
    
    def benchmark_hp_inv_vs_numpy(self, matrix_size: int, num_trials: int = 5) -> List[BenchmarkResult]:
        """Benchmark HP-INV against NumPy's direct solver."""
        print(f"Benchmarking HP-INV vs NumPy for {matrix_size}x{matrix_size} matrices ({num_trials} trials)...")
        
        results = []
        
        for trial in range(num_trials):
            # Create a well-conditioned matrix
            A = np.random.rand(matrix_size, matrix_size) * 0.1 + np.eye(matrix_size)
            b = np.random.rand(matrix_size)
            
            # Benchmark NumPy direct solver
            start_time = time.time()
            x_numpy = np.linalg.solve(A, b)
            numpy_time = time.time() - start_time
            numpy_residual = np.linalg.norm(A @ x_numpy - b)
            
            # Benchmark HP-INV
            start_time = time.time()
            x_hp, iters, info = hp_inv(A, b, bits=4, max_iter=20, tol=1e-6)
            hp_time = time.time() - start_time
            hp_residual = np.linalg.norm(A @ x_hp - b)
            
            # Calculate errors
            numpy_error = np.linalg.norm(x_numpy - x_hp) / np.linalg.norm(x_numpy)
            
            # Store results
            results.append(BenchmarkResult(
                name=f"NumPy_{matrix_size}x{matrix_size}_trial_{trial}",
                matrix_size=matrix_size,
                time_elapsed=numpy_time,
                residual=numpy_residual,
                iterations=1,  # Direct solver
                error=0.0,  # NumPy is our reference
                condition_number=np.linalg.cond(A),
                additional_info={'method': 'numpy_direct'}
            ))
            
            results.append(BenchmarkResult(
                name=f"HP_INV_{matrix_size}x{matrix_size}_trial_{trial}",
                matrix_size=matrix_size,
                time_elapsed=hp_time,
                residual=hp_residual,
                iterations=iters,
                error=numpy_error,
                condition_number=np.linalg.cond(A),
                additional_info={'method': 'hp_inv', 'converged': info['converged']}
            ))
        
        self.results.extend(results)
        return results
    
    def benchmark_rram_effects(self, matrix_size: int, num_trials: int = 3) -> List[BenchmarkResult]:
        """Benchmark how RRAM effects impact performance."""
        print(f"Benchmarking RRAM effects for {matrix_size}x{matrix_size} matrices...")
        
        results = []
        
        for trial in range(num_trials):
            # Create matrix with RRAM effects
            A_rram = create_rram_matrix(
                matrix_size, 
                variability=0.03, 
                stuck_fault_prob=0.02,
                line_resistance=1.5e-3
            )
            A_rram = A_rram + 0.3 * np.eye(matrix_size)  # Ensure conditioning
            A_rram = apply_redundancy(A_rram)
            
            b = np.random.rand(matrix_size)
            
            # Test standard solver on RRAM matrix
            start_time = time.time()
            x_standard, std_iters, std_info = hp_inv(A_rram, b, bits=3, max_iter=25, tol=1e-5)
            std_time = time.time() - start_time
            std_residual = np.linalg.norm(A_rram @ x_standard - b)
            
            # Test adaptive solver on RRAM matrix
            start_time = time.time()
            x_adaptive, adapt_iters, adapt_info = adaptive_hp_inv(
                A_rram, b, initial_bits=2, max_bits=5, max_iter=25, tol=1e-5
            )
            adapt_time = time.time() - start_time
            adapt_residual = np.linalg.norm(A_rram @ x_adaptive - b)
            
            results.append(BenchmarkResult(
                name=f"Standard_HP_INV_RRAM_{matrix_size}x{matrix_size}_trial_{trial}",
                matrix_size=matrix_size,
                time_elapsed=std_time,
                residual=std_residual,
                iterations=std_iters,
                error=0.0,  # Will be calculated relative to each other
                condition_number=np.linalg.cond(A_rram),
                additional_info={'method': 'standard_hp_inv', 'converged': std_info['converged']}
            ))
            
            results.append(BenchmarkResult(
                name=f"Adaptive_HP_INV_RRAM_{matrix_size}x{matrix_size}_trial_{trial}",
                matrix_size=matrix_size,
                time_elapsed=adapt_time,
                residual=adapt_residual,
                iterations=adapt_iters,
                error=0.0,  # Will be calculated relative to each other
                condition_number=np.linalg.cond(A_rram),
                additional_info={'method': 'adaptive_hp_inv', 'converged': adapt_info['converged'], 
                               'final_bits': adapt_info.get('final_bits', 2)}
            ))
        
        self.results.extend(results)
        return results
    
    def benchmark_scalability(self, sizes: List[int], num_trials: int = 3) -> List[BenchmarkResult]:
        """Benchmark scalability across different matrix sizes."""
        print(f"Benchmarking scalability for sizes: {sizes}")
        
        results = []
        
        for size in sizes:
            for trial in range(num_trials):
                # Create a well-conditioned matrix
                A = np.random.rand(size, size) * 0.1 + np.eye(size)
                b = np.random.rand(size)
                
                # Test standard HP-INV
                start_time = time.time()
                x_std, iters_std, info_std = hp_inv(A, b, bits=4, max_iter=20, tol=1e-6)
                std_time = time.time() - start_time
                std_residual = np.linalg.norm(A @ x_std - b)
                
                # Test block HP-INV for larger sizes
                if size >= 8:
                    start_time = time.time()
                    x_block, iters_block, info_block = block_hp_inv(
                        A, b, block_size=size//2, max_iter=20, tol=1e-6
                    )
                    block_time = time.time() - start_time
                    block_residual = np.linalg.norm(A @ x_block - b)
                    
                    results.append(BenchmarkResult(
                        name=f"Block_HP_INV_{size}x{size}_trial_{trial}",
                        matrix_size=size,
                        time_elapsed=block_time,
                        residual=block_residual,
                        iterations=iters_block,
                        error=np.linalg.norm(x_block - x_std) / np.linalg.norm(x_std),
                        condition_number=np.linalg.cond(A),
                        additional_info={'method': 'block_hp_inv', 'converged': info_block['converged']}
                    ))
                
                results.append(BenchmarkResult(
                    name=f"Standard_HP_INV_{size}x{size}_trial_{trial}",
                    matrix_size=size,
                    time_elapsed=std_time,
                    residual=std_residual,
                    iterations=iters_std,
                    error=0.0,  # Reference
                    condition_number=np.linalg.cond(A),
                    additional_info={'method': 'standard_hp_inv', 'converged': info_std['converged']}
                ))
        
        self.results.extend(results)
        return results
    
    def benchmark_blockamc_vs_direct(self, matrix_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark BlockAMC algorithms vs direct methods."""
        print(f"Benchmarking BlockAMC vs Direct for sizes: {matrix_sizes}")
        
        results = []
        
        for size in matrix_sizes:
            # Create a well-conditioned matrix for inversion
            A = np.random.rand(size, size) * 0.1 + np.eye(size)
            
            # Benchmark direct inversion
            start_time = time.time()
            A_inv_direct = np.linalg.inv(A)
            direct_time = time.time() - start_time
            direct_error = np.linalg.norm(A @ A_inv_direct - np.eye(size))
            
            # Benchmark BlockAMC inversion
            block_size = min(8, size)  # Use 8 as max block size
            if size > 4:  # Only use BlockAMC for larger matrices
                start_time = time.time()
                A_inv_block = blockamc_inversion(A, block_size=block_size)
                block_time = time.time() - start_time
                block_error = np.linalg.norm(A @ A_inv_block - np.eye(size))
                
                results.append(BenchmarkResult(
                    name=f"BlockAMC_Inversion_{size}x{size}",
                    matrix_size=size,
                    time_elapsed=block_time,
                    residual=0.0,  # Inversion benchmark
                    iterations=0,
                    error=block_error,
                    condition_number=np.linalg.cond(A),
                    additional_info={'method': 'blockamc_inversion'}
                ))
            
            results.append(BenchmarkResult(
                name=f"Direct_Inversion_{size}x{size}",
                matrix_size=size,
                time_elapsed=direct_time,
                residual=0.0,  # Inversion benchmark
                iterations=0,
                error=direct_error,
                condition_number=np.linalg.cond(A),
                additional_info={'method': 'direct_inversion'}
            ))
        
        self.results.extend(results)
        return results
    
    def generate_performance_report(self) -> str:
        """Generate a performance report from the collected benchmarks."""
        if not self.results:
            return "No benchmark results collected yet."
        
        report = []
        report.append("Analogico Project - Performance Benchmark Report\n")
        report.append("=" * 50)
        
        # Group results by method
        methods = {}
        for result in self.results:
            method = result.additional_info.get('method', 'unknown') if result.additional_info else 'unknown'
            if method not in methods:
                methods[method] = []
            methods[method].append(result)
        
        for method, method_results in methods.items():
            report.append(f"\n{method.upper()} Performance:")
            report.append("-" * 30)
            
            times = [r.time_elapsed for r in method_results]
            residuals = [r.residual for r in method_results]
            errors = [r.error for r in method_results]
            iters = [r.iterations for r in method_results if r.iterations > 0]
            
            report.append(f"  Average Time: {np.mean(times):.4f}s (std: {np.std(times):.4f})")
            report.append(f"  Average Residual: {np.mean(residuals):.2e}")
            report.append(f"  Average Error: {np.mean(errors):.2e}")
            if iters:
                report.append(f"  Average Iterations: {np.mean(iters):.1f}")
        
        # Compare methods
        if 'numpy_direct' in methods and 'hp_inv' in methods:
            numpy_times = [r.time_elapsed for r in methods['numpy_direct']]
            hp_times = [r.time_elapsed for r in methods['hp_inv'] if r.additional_info.get('method') == 'hp_inv']
            
            if numpy_times and hp_times:
                avg_speedup = np.mean(numpy_times) / np.mean(hp_times)
                report.append(f"\nHP-INV vs NumPy Speed Comparison:")
                report.append(f"  Average Speedup: {avg_speedup:.2f}x (HP-INV faster if > 1)")
        
        return "\n".join(report)
    
    def plot_benchmark_results(self):
        """Plot the benchmark results."""
        if not self.results:
            print("No results to plot")
            return
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        methods = []
        times = []
        residuals = []
        matrix_sizes = []
        
        for result in self.results:
            method = result.additional_info.get('method', 'unknown') if result.additional_info else 'unknown'
            methods.append(method)
            times.append(result.time_elapsed)
            residuals.append(result.residual)
            matrix_sizes.append(result.matrix_size)
        
        # Convert to numpy arrays for easier manipulation
        methods = np.array(methods)
        times = np.array(times)
        residuals = np.array(residuals)
        matrix_sizes = np.array(matrix_sizes)
        
        # Plot 1: Time by method
        unique_methods = np.unique(methods)
        for method in unique_methods:
            mask = methods == method
            avg_time = np.mean(times[mask]) if np.any(mask) else 0
            ax1.bar(method, avg_time, label=method, alpha=0.7)
        
        ax1.set_ylabel('Average Time (s)')
        ax1.set_title('Average Computation Time by Method')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Residual by method
        for method in unique_methods:
            mask = methods == method
            avg_residual = np.mean(residuals[mask]) if np.any(mask) else 0
            ax2.bar(method, avg_residual, label=method, alpha=0.7)
        
        ax2.set_ylabel('Average Residual')
        ax2.set_title('Average Residual by Method')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yscale('log')
        
        # Plot 3: Time vs matrix size for standard HP-INV
        hp_inv_mask = (methods == 'standard_hp_inv')
        if np.any(hp_inv_mask):
            ax3.scatter(matrix_sizes[hp_inv_mask], times[hp_inv_mask], 
                       label='Standard HP-INV', alpha=0.7)
        
        block_mask = (methods == 'block_hp_inv')
        if np.any(block_mask):
            ax3.scatter(matrix_sizes[block_mask], times[block_mask], 
                       label='Block HP-INV', alpha=0.7)
        
        ax3.set_xlabel('Matrix Size')
        ax3.set_ylabel('Time (s)')
        ax3.set_title('Computation Time vs Matrix Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Convergence behavior (iterations vs size for HP-INV methods)
        hp_inv_results = [r for r in self.results 
                         if 'hp_inv' in r.name and r.iterations > 0]
        if hp_inv_results:
            sizes = [r.matrix_size for r in hp_inv_results]
            iters = [r.iterations for r in hp_inv_results]
            
            ax4.scatter(sizes, iters, alpha=0.7)
            ax4.set_xlabel('Matrix Size')
            ax4.set_ylabel('Iterations to Convergence')
            ax4.set_title('Iterations vs Matrix Size')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def run_comprehensive_benchmarks():
    """Run a comprehensive set of benchmarks."""
    print("Starting comprehensive benchmarking...")
    
    benchmark_suite = BenchmarkSuite()
    
    # Run different benchmark sets
    print("\n1. HP-INV vs NumPy comparison...")
    benchmark_suite.benchmark_hp_inv_vs_numpy(matrix_size=8, num_trials=3)
    
    print("\n2. RRAM effects benchmark...")
    benchmark_suite.benchmark_rram_effects(matrix_size=6, num_trials=2)
    
    print("\n3. Scalability benchmark...")
    benchmark_suite.benchmark_scalability(sizes=[4, 8, 12], num_trials=2)
    
    print("\n4. BlockAMC vs Direct inversion...")
    benchmark_suite.benchmark_blockamc_vs_direct(matrix_sizes=[6, 8, 10])
    
    # Generate report
    print("\n" + "="*60)
    print("BENCHMARKING COMPLETE")
    print("="*60)
    print(benchmark_suite.generate_performance_report())
    
    # Optionally plot results
    try:
        benchmark_suite.plot_benchmark_results()
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    return benchmark_suite


def benchmark_memory_usage():
    """Benchmark memory usage of different approaches."""
    import psutil
    import gc
    
    print("Memory usage benchmarking...")
    
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Test memory usage for different matrix sizes
    sizes = [16, 32, 64]
    
    results = []
    for size in sizes:
        print(f"Benchmarking memory usage for {size}x{size} matrix...")
        
        # Create matrix
        base_memory = get_memory_usage()
        A = np.random.rand(size, size) * 0.1 + np.eye(size)
        b = np.random.rand(size)
        matrix_memory = get_memory_usage() - base_memory
        
        # Test NumPy direct solve memory
        gc.collect()
        start_memory = get_memory_usage()
        x_numpy = np.linalg.solve(A, b)
        numpy_memory = get_memory_usage() - start_memory
        
        # Test HP-INV memory
        gc.collect()
        start_memory = get_memory_usage()
        x_hp, iters, info = hp_inv(A, b, bits=4, max_iter=20)
        hp_memory = get_memory_usage() - start_memory
        
        results.append({
            'size': size,
            'matrix_memory_mb': matrix_memory,
            'numpy_memory_mb': numpy_memory,
            'hp_memory_mb': hp_memory
        })
        
        print(f"  Matrix: {matrix_memory:.2f}MB, NumPy: {numpy_memory:.2f}MB, HP-INV: {hp_memory:.2f}MB")
    
    return results


if __name__ == "__main__":
    # Run comprehensive benchmarks
    benchmark_suite = run_comprehensive_benchmarks()
    
    # Run memory usage benchmark
    try:
        mem_results = benchmark_memory_usage()
        print("\nMemory Usage Results:")
        for result in mem_results:
            print(f"Size {result['size']}x{result['size']}: Matrix={result['matrix_memory_mb']:.2f}MB, "
                  f"NumPy={result['numpy_memory_mb']:.2f}MB, HP-INV={result['hp_memory_mb']:.2f}MB")
    except ImportError:
        print("psutil not available, skipping memory benchmark")