"""
Benchmarking Suite for RRAM-based Computing Systems.

This module provides comprehensive benchmarking tools to compare
RRAM-based implementations against digital alternatives across
multiple metrics and use cases.
"""
import time
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
import json

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV
    from .neuromorphic_modules import RRAMNeuromorphicNetwork, RRAMSpikingLayer, RRAMReservoirComputer
    from .model_deployment_pipeline import benchmark_rram_vs_digital
    from .optimization_algorithms import RRAMAwareOptimizer, OptimizationTarget
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class BenchmarkType(Enum):
    """Types of benchmarks available."""
    LINEAR_SOLVER = "linear_solver"
    NEURAL_NETWORK = "neural_network"
    MATRIX_OPERATIONS = "matrix_operations"
    POWER_EFFICIENCY = "power_efficiency"
    STABILITY = "stability"
    THROUGHPUT = "throughput"


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    benchmark_type: BenchmarkType
    metric_name: str
    value: float
    unit: str
    timestamp: float
    configuration: Dict[str, Any]
    notes: Optional[str] = None


class RRAMBenchmarkSuite:
    """
    Comprehensive benchmarking suite for RRAM-based systems.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 save_results: bool = True,
                 results_dir: str = "benchmark_results"):
        """
        Initialize the benchmark suite.
        
        Args:
            rram_interface: Optional RRAM interface for hardware benchmarks
            save_results: Whether to save results to files
            results_dir: Directory to save results
        """
        self.rram_interface = rram_interface
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.benchmark_results = []
        self.benchmark_history = {}
    
    def run_linear_solver_benchmark(
        self, 
        matrix_sizes: List[int], 
        matrix_types: Optional[List[str]] = None,
        num_samples: int = 5,
        include_digital: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark linear solver performance across different matrix sizes and types.
        
        Args:
            matrix_sizes: List of matrix sizes to test
            matrix_types: Types of matrices ('random', 'diagonally_dominant', 'ill_conditioned', etc.)
            num_samples: Number of samples per size/type
            include_digital: Whether to include digital benchmarks for comparison
            
        Returns:
            Dictionary of benchmark results
        """
        if matrix_types is None:
            matrix_types = ['random', 'diagonally_dominant', 'ill_conditioned']
        
        results = {
            'rram_performance': [],
            'digital_performance': [] if include_digital else [],
            'matrix_sizes': matrix_sizes,
            'matrix_types': matrix_types
        }
        
        for matrix_size in matrix_sizes:
            for matrix_type in matrix_types:
                for sample_idx in range(num_samples):
                    # Generate test matrix
                    G, b = self._generate_test_matrix(matrix_size, matrix_type)
                    
                    # Benchmark RRAM-based solver
                    rram_start = time.time()
                    try:
                        if self.rram_interface:
                            # Use hardware interface if available
                            self.rram_interface.write_matrix(G)
                            x_rram, iter_rram, info_rram = hp_inv(G, b)
                            rram_time = time.time() - rram_start
                        else:
                            # Use simulation
                            x_rram, iter_rram, info_rram = hp_inv(G, b)
                            rram_time = time.time() - rram_start
                    except Exception as e:
                        rram_time = float('inf')
                        warnings.warn(f"RRAM benchmark failed for size {matrix_size}: {e}")
                        continue
                    
                    # Benchmark digital solver if requested
                    if include_digital:
                        digital_start = time.time()
                        try:
                            x_digital = np.linalg.solve(G, b)
                            digital_time = time.time() - digital_start
                        except Exception as e:
                            digital_time = float('inf')
                            warnings.warn(f"Digital benchmark failed for size {matrix_size}: {e}")
                            continue
                    else:
                        digital_time = 0.0
                    
                    # Calculate accuracy
                    rram_accuracy = np.linalg.norm(G @ x_rram - b) if rram_time != float('inf') else float('inf')
                    digital_accuracy = np.linalg.norm(G @ x_digital - b) if include_digital else 0.0
                    
                    # Store results
                    rram_result = {
                        'size': matrix_size,
                        'type': matrix_type,
                        'sample': sample_idx,
                        'time': rram_time,
                        'accuracy': rram_accuracy,
                        'iterations': iter_rram if rram_time != float('inf') else 0,
                        'method': 'rram'
                    }
                    results['rram_performance'].append(rram_result)
                    
                    if include_digital:
                        digital_result = {
                            'size': matrix_size,
                            'type': matrix_type,
                            'sample': sample_idx,
                            'time': digital_time,
                            'accuracy': digital_accuracy,
                            'method': 'digital'
                        }
                        results['digital_performance'].append(digital_result)
        
        # Add to benchmark history
        self.benchmark_history['linear_solver'] = results
        
        return results
    
    def run_neural_network_benchmark(
        self,
        network_configs: List[Dict[str, Any]],
        num_samples: int = 10,
        include_digital: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark neural network performance using RRAM.
        
        Args:
            network_configs: List of network configurations to test
            num_samples: Number of samples per configuration
            include_digital: Whether to include digital benchmarks for comparison
            
        Returns:
            Dictionary of benchmark results
        """
        from .ml_framework_integration import TorchRRAMLinear, create_rram_neural_network
        
        if not CORE_MODULES_AVAILABLE:
            warnings.warn("Core modules not available for neural network benchmarking")
            return {}
        
        results = {
            'rram_network_performance': [],
            'digital_network_performance': [] if include_digital else [],
            'configurations': network_configs
        }
        
        for config in network_configs:
            for sample_idx in range(num_samples):
                # Create RRAM neural network
                try:
                    input_size = config.get('input_size', 10)
                    hidden_sizes = config.get('hidden_sizes', [20, 15])
                    output_size = config.get('output_size', 1)
                    
                    rram_network = create_rram_neural_network(
                        input_size, hidden_sizes, output_size,
                        rram_interface=self.rram_interface
                    )
                    
                    # Generate test data
                    test_input = torch.randn(1, input_size)
                    
                    # Benchmark RRAM network
                    rram_start = time.time()
                    with torch.no_grad():
                        rram_output = rram_network(test_input)
                    rram_time = time.time() - rram_start
                    
                    # Benchmark digital network for comparison
                    if include_digital:
                        import torch.nn as nn
                        digital_network = nn.Sequential()
                        layer_sizes = [input_size] + hidden_sizes + [output_size]
                        for i in range(len(layer_sizes) - 1):
                            digital_network.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                            if i < len(layer_sizes) - 2:
                                digital_network.append(nn.ReLU())
                        
                        digital_start = time.time()
                        with torch.no_grad():
                            digital_output = digital_network(test_input)
                        digital_time = time.time() - digital_start
                    else:
                        digital_time = 0.0
                    
                    # Store results
                    rram_result = {
                        'config': config,
                        'sample': sample_idx,
                        'time': rram_time,
                        'method': 'rram_neural_network'
                    }
                    results['rram_network_performance'].append(rram_result)
                    
                    if include_digital:
                        digital_result = {
                            'config': config,
                            'sample': sample_idx,
                            'time': digital_time,
                            'method': 'digital_neural_network'
                        }
                        results['digital_network_performance'].append(digital_result)
                        
                except Exception as e:
                    warnings.warn(f"Neural network benchmark failed: {e}")
                    continue
        
        # Add to benchmark history
        self.benchmark_history['neural_network'] = results
        
        return results
    
    def run_power_efficiency_benchmark(
        self,
        matrix_sizes: List[int],
        duration: float = 10.0  # seconds
    ) -> Dict[str, Any]:
        """
        Benchmark power efficiency of RRAM vs digital implementations.
        
        Args:
            matrix_sizes: List of matrix sizes to test
            duration: Duration to run continuous operations (in seconds)
            
        Returns:
            Dictionary of power efficiency results
        """
        results = {
            'rram_power_metrics': [],
            'digital_power_metrics': [],
            'matrix_sizes': matrix_sizes
        }
        
        # This is a simulation - in real hardware, you would measure actual power
        for size in matrix_sizes:
            # For simulation, estimate based on operations
            num_ops_rram = 0
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Generate test matrix
                G, b = self._generate_test_matrix(size, 'random')
                
                # Perform operation
                try:
                    x, iters, info = hp_inv(G, b)
                    num_ops_rram += 1
                except:
                    continue
            
            actual_duration = time.time() - start_time
            ops_per_second_rram = num_ops_rram / actual_duration if actual_duration > 0 else 0
            
            # Estimate power (this is simulated - real implementation would measure actual power)
            estimated_power_rram = 1e-6 * size * size * 1e-3  # Simplified model
            energy_per_op_rram = estimated_power_rram / ops_per_second_rram if ops_per_second_rram > 0 else float('inf')
            
            rram_metrics = {
                'size': size,
                'operations_per_second': ops_per_second_rram,
                'estimated_power_watts': estimated_power_rram,
                'energy_per_operation_joules': energy_per_op_rram,
                'duration_seconds': actual_duration,
                'total_operations': num_ops_rram
            }
            results['rram_power_metrics'].append(rram_metrics)
            
            # For digital comparison, estimate higher power
            estimated_power_digital = 1e-3 * size * size * 1e-3  # Higher power for digital
            energy_per_op_digital = estimated_power_digital / ops_per_second_rram if ops_per_second_rram > 0 else float('inf')
            
            digital_metrics = {
                'size': size,
                'operations_per_second': ops_per_second_rram,
                'estimated_power_watts': estimated_power_digital,
                'energy_per_operation_joules': energy_per_op_digital,
                'duration_seconds': actual_duration,
                'total_operations': num_ops_rram
            }
            results['digital_power_metrics'].append(digital_metrics)
        
        # Add to benchmark history
        self.benchmark_history['power_efficiency'] = results
        
        return results
    
    def run_stability_benchmark(
        self,
        matrix_size: int = 100,
        num_iterations: int = 100,
        perturbation_magnitude: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Benchmark the stability of RRAM-based computations.
        
        Args:
            matrix_size: Size of matrix to use for stability testing
            num_iterations: Number of iterations to test stability
            perturbation_magnitude: Magnitude of perturbations to test
            
        Returns:
            Dictionary of stability benchmark results
        """
        results = {
            'stability_metrics': [],
            'matrix_size': matrix_size,
            'num_iterations': num_iterations
        }
        
        # Generate a base matrix and vector
        G_base, b_base = self._generate_test_matrix(matrix_size, 'random')
        
        # Store initial solution
        x_init, _, _ = hp_inv(G_base, b_base)
        
        deviations = []
        for i in range(num_iterations):
            # Add small perturbation to matrix
            G_perturbed = G_base + np.random.normal(0, perturbation_magnitude, G_base.shape)
            b_perturbed = b_base + np.random.normal(0, perturbation_magnitude, b_base.shape)
            
            try:
                x_perturbed, _, _ = hp_inv(G_perturbed, b_perturbed)
                deviation = np.linalg.norm(x_perturbed - x_init)
                deviations.append(deviation)
            except:
                deviations.append(float('inf'))
        
        # Calculate stability metrics
        stability_metrics = {
            'mean_deviation': float(np.mean(deviations)),
            'std_deviation': float(np.std(deviations)),
            'max_deviation': float(np.max(deviations)),
            'min_deviation': float(np.min(deviations)),
            'stability_index': float(1.0 / (1.0 + np.std(deviations) + np.mean(deviations))),  # Lower is better
            'condition_number': float(np.linalg.cond(G_base))
        }
        
        results['stability_metrics'] = stability_metrics
        
        # Add to benchmark history
        self.benchmark_history['stability'] = results
        
        return results
    
    def run_throughput_benchmark(
        self,
        matrix_sizes: List[int],
        duration: float = 10.0  # seconds
    ) -> Dict[str, Any]:
        """
        Benchmark computational throughput of RRAM vs digital implementations.
        
        Args:
            matrix_sizes: List of matrix sizes to test
            duration: Duration to run throughput test (in seconds)
            
        Returns:
            Dictionary of throughput results
        """
        results = {
            'rram_throughput': [],
            'digital_throughput': [],
            'matrix_sizes': matrix_sizes
        }
        
        for size in matrix_sizes:
            # Test RRAM throughput
            num_completed_rram = 0
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Generate test problem
                G, b = self._generate_test_matrix(size, 'random')
                
                # Attempt to solve
                try:
                    x, iters, info = hp_inv(G, b)
                    num_completed_rram += 1
                except:
                    continue
            
            actual_duration = time.time() - start_time
            throughput_rram = num_completed_rram / actual_duration if actual_duration > 0 else 0
            
            rram_result = {
                'size': size,
                'operations_completed': num_completed_rram,
                'duration': actual_duration,
                'throughput_ops_per_sec': throughput_rram,
                'method': 'rram'
            }
            results['rram_throughput'].append(rram_result)
            
            # Test digital throughput
            if CORE_MODULES_AVAILABLE:
                num_completed_digital = 0
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    # Generate test problem
                    G, b = self._generate_test_matrix(size, 'random')
                    
                    # Attempt to solve digitally
                    try:
                        x = np.linalg.solve(G, b)
                        num_completed_digital += 1
                    except:
                        continue
                
                actual_duration = time.time() - start_time
                throughput_digital = num_completed_digital / actual_duration if actual_duration > 0 else 0
                
                digital_result = {
                    'size': size,
                    'operations_completed': num_completed_digital,
                    'duration': actual_duration,
                    'throughput_ops_per_sec': throughput_digital,
                    'method': 'digital'
                }
                results['digital_throughput'].append(digital_result)
        
        # Add to benchmark history
        self.benchmark_history['throughput'] = results
        
        return results
    
    def _generate_test_matrix(self, size: int, matrix_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate test matrix of specified type and size.
        
        Args:
            size: Size of the square matrix
            matrix_type: Type of matrix to generate
            
        Returns:
            Tuple of (matrix G, vector b)
        """
        if matrix_type == 'random':
            G = np.random.rand(size, size) * 2 - 1  # Values between -1 and 1
            # Ensure diagonal dominance to make it well-conditioned
            G = G + size * np.eye(size)
        elif matrix_type == 'diagonally_dominant':
            G = np.random.rand(size, size) * 0.5
            # Ensure diagonal dominance
            for i in range(size):
                row_sum = np.sum(np.abs(G[i, :])) - np.abs(G[i, i])
                G[i, i] = row_sum + 0.1  # Make diagonally dominant
        elif matrix_type == 'ill_conditioned':
            # Create an ill-conditioned matrix using Hilbert-like structure
            G = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    G[i, j] = 1.0 / (i + j + 1)
            # Add small random noise
            G = G + 1e-10 * np.random.rand(size, size)
        elif matrix_type == 'symmetric_positive_definite':
            # Generate a random matrix and make it symmetric positive definite
            A = np.random.rand(size, size)
            G = A @ A.T  # A*A^T is symmetric positive definite
        else:  # Default to random
            G = np.random.rand(size, size)
        
        # Generate random b vector
        b = np.random.rand(size)
        
        return G, b
    
    def generate_benchmark_report(self, 
                                 report_format: str = 'text',
                                 benchmark_types: Optional[List[BenchmarkType]] = None) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            report_format: Format of the report ('text', 'markdown', 'json')
            benchmark_types: Specific benchmark types to include (None for all)
            
        Returns:
            Benchmark report as string
        """
        if benchmark_types is None:
            benchmark_types = [bt for bt in BenchmarkType]
        
        if report_format == 'json':
            return json.dumps(self.benchmark_history, indent=2, default=str)
        elif report_format == 'markdown':
            report = "# RRAM Benchmark Report\n\n"
            report += f"Generated on: {time.ctime()}\n\n"
            
            for benchmark_type in benchmark_types:
                type_str = benchmark_type.value
                if type_str in self.benchmark_history:
                    report += f"## {type_str.replace('_', ' ').title()} Benchmarks\n\n"
                    results = self.benchmark_history[type_str]
                    
                    # Add summary statistics for this benchmark type
                    if type_str == 'linear_solver':
                        if 'rram_performance' in results:
                            avg_time = np.mean([r['time'] for r in results['rram_performance']])
                            avg_accuracy = np.mean([r['accuracy'] for r in results['rram_performance']])
                            report += f"- Average RRAM time: {avg_time:.6f}s\n"
                            report += f"- Average RRAM accuracy: {avg_accuracy:.2e}\n"
                    elif type_str == 'throughput':
                        if 'rram_throughput' in results:
                            avg_throughput = np.mean([r['throughput_ops_per_sec'] for r in results['rram_throughput']])
                            report += f"- Average RRAM throughput: {avg_throughput:.2f} ops/s\n"
            
            return report
        else:  # Default to text
            report = "RRAM Benchmark Report\n"
            report += "=" * 50 + "\n"
            report += f"Generated on: {time.ctime()}\n\n"
            
            for benchmark_type in benchmark_types:
                type_str = benchmark_type.value
                if type_str in self.benchmark_history:
                    report += f"{type_str.replace('_', ' ').title()} Benchmarks:\n"
                    report += "-" * 30 + "\n"
                    
                    results = self.benchmark_history[type_str]
                    # Add more detailed information based on result type
                    if type_str == 'linear_solver':
                        if 'rram_performance' in results and results['rram_performance']:
                            avg_time = np.mean([r['time'] for r in results['rram_performance']])
                            avg_accuracy = np.mean([r['accuracy'] for r in results['rram_performance']])
                            report += f"  Avg RRAM time: {avg_time:.6f}s\n"
                            report += f"  Avg RRAM accuracy: {avg_accuracy:.2e}\n"
                    elif type_str == 'throughput':
                        if 'rram_throughput' in results and results['rram_throughput']:
                            avg_throughput = np.mean([r['throughput_ops_per_sec'] for r in results['rram_throughput']])
                            report += f"  Avg RRAM throughput: {avg_throughput:.2f} ops/s\n"
                    
                    report += "\n"
            
            return report
    
    def save_benchmark_results(self, filename: str = "benchmark_results") -> str:
        """
        Save benchmark results to file.
        
        Args:
            filename: Name of file to save results (without extension)
            
        Returns:
            Path to saved file
        """
        # Save as JSON
        json_path = self.results_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(self.benchmark_history, f, indent=2, default=str)
        
        # Also save as CSV where possible
        for bench_type, results in self.benchmark_history.items():
            if isinstance(results, dict) and any(isinstance(v, list) for v in results.values()):
                # Find the first list value to convert to DataFrame
                for key, value in results.items():
                    if isinstance(value, list) and len(value) > 0:
                        try:
                            df = pd.DataFrame(value)
                            csv_path = self.results_dir / f"{filename}_{bench_type}_{key}.csv"
                            df.to_csv(csv_path, index=False)
                        except Exception as e:
                            warnings.warn(f"Could not save {bench_type} {key} to CSV: {e}")
        
        return str(json_path)


class BenchmarkPlotter:
    """
    Tools for creating visualizations of benchmark results.
    """
    
    def __init__(self):
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_linear_solver_performance(self, 
                                     benchmark_results: Dict[str, Any],
                                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot linear solver performance comparison.
        
        Args:
            benchmark_results: Results from linear solver benchmark
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract data
        rram_data = benchmark_results.get('rram_performance', [])
        digital_data = benchmark_results.get('digital_performance', [])
        
        if not rram_data:
            fig.suptitle("No RRAM data available")
            return fig
        
        # Extract metrics
        rram_sizes = [r['size'] for r in rram_data]
        rram_times = [r['time'] for r in rram_data]
        rram_accuracies = [r['accuracy'] for r in rram_data]
        
        digital_sizes = [r['size'] for r in digital_data]
        digital_times = [r['time'] for r in digital_data]
        digital_accuracies = [r['accuracy'] for r in digital_data]
        
        # Plot 1: Time vs Size
        if digital_times:
            axes[0, 0].scatter(digital_sizes, digital_times, label='Digital', alpha=0.7)
        axes[0, 0].scatter(rram_sizes, rram_times, label='RRAM', alpha=0.7)
        axes[0, 0].set_xlabel('Matrix Size')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].set_title('Time vs Matrix Size')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # Plot 2: Accuracy vs Size
        if digital_accuracies:
            axes[0, 1].scatter(digital_sizes, digital_accuracies, label='Digital', alpha=0.7)
        axes[0, 1].scatter(rram_sizes, rram_accuracies, label='RRAM', alpha=0.7)
        axes[0, 1].set_xlabel('Matrix Size')
        axes[0, 1].set_ylabel('Accuracy (Residual Norm)')
        axes[0, 1].set_title('Accuracy vs Matrix Size')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Time comparison by matrix type
        if rram_data:
            df_rram = pd.DataFrame(rram_data)
            df_digital = pd.DataFrame(digital_data) if digital_data else pd.DataFrame()
            
            if 'type' in df_rram.columns:
                # Group by matrix type and calculate average time
                rram_avg_time = df_rram.groupby('type')['time'].mean()
                if not df_digital.empty and 'type' in df_digital.columns:
                    digital_avg_time = df_digital.groupby('type')['time'].mean()
                    
                    x = range(len(rram_avg_time))
                    width = 0.35
                    
                    axes[1, 0].bar([i - width/2 for i in x], rram_avg_time.values, 
                                 width, label='RRAM', alpha=0.7)
                    axes[1, 0].bar([i + width/2 for i in x], digital_avg_time.values, 
                                 width, label='Digital', alpha=0.7)
                    axes[1, 0].set_xlabel('Matrix Type')
                    axes[1, 0].set_ylabel('Average Time (s)')
                    axes[1, 0].set_title('Average Time by Matrix Type')
                    axes[1, 0].set_xticks(x)
                    axes[1, 0].set_xticklabels(rram_avg_time.index, rotation=45)
                    axes[1, 0].legend()
                    axes[1, 0].set_yscale('log')
        
        # Plot 4: Efficiency (accuracy/time) comparison
        if digital_times and digital_accuracies:
            digital_efficiency = [acc/t if t > 0 else 0 for acc, t in zip(digital_accuracies, digital_times)]
        else:
            digital_efficiency = []
        
        rram_efficiency = [acc/t if t > 0 else 0 for acc, t in zip(rram_accuracies, rram_times)]
        
        axes[1, 1].scatter(range(len(digital_efficiency)), digital_efficiency, 
                         label='Digital', alpha=0.7, marker='s')
        axes[1, 1].scatter(range(len(rram_efficiency)), rram_efficiency, 
                         label='RRAM', alpha=0.7)
        axes[1, 1].set_xlabel('Test Case')
        axes[1, 1].set_ylabel('Efficiency (Accuracy/Time)')
        axes[1, 1].set_title('Efficiency Comparison')
        axes[1, 1].legend()
        
        fig.tight_layout()
        return fig
    
    def plot_power_efficiency(self,
                            benchmark_results: Dict[str, Any],
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot power efficiency comparison.
        
        Args:
            benchmark_results: Results from power efficiency benchmark
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        rram_metrics = benchmark_results.get('rram_power_metrics', [])
        digital_metrics = benchmark_results.get('digital_power_metrics', [])
        
        if not rram_metrics:
            ax.text(0.5, 0.5, "No power efficiency data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            return fig
        
        # Extract data
        sizes = [m['size'] for m in rram_metrics]
        rram_energy_per_op = [m['energy_per_operation_joules'] for m in rram_metrics]
        digital_energy_per_op = [m['energy_per_operation_joules'] for m in digital_metrics]
        
        # Plot energy per operation
        ax.scatter(sizes, digital_energy_per_op, label='Digital', alpha=0.7, marker='s')
        ax.scatter(sizes, rram_energy_per_op, label='RRAM', alpha=0.7)
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Energy per Operation (J)')
        ax.set_title('Energy Efficiency Comparison')
        ax.legend()
        ax.set_yscale('log')
        
        return fig
    
    def plot_throughput(self,
                       benchmark_results: Dict[str, Any],
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot throughput comparison.
        
        Args:
            benchmark_results: Results from throughput benchmark
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        rram_results = benchmark_results.get('rram_throughput', [])
        digital_results = benchmark_results.get('digital_throughput', [])
        
        if not rram_results:
            ax.text(0.5, 0.5, "No throughput data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            return fig
        
        # Extract data
        sizes = [r['size'] for r in rram_results]
        rram_throughput = [r['throughput_ops_per_sec'] for r in rram_results]
        digital_throughput = [r['throughput_ops_per_sec'] for r in digital_results]
        
        # Plot throughput comparison
        x = np.arange(len(sizes))
        width = 0.35
        
        ax.bar(x - width/2, digital_throughput, width, label='Digital', alpha=0.7)
        ax.bar(x + width/2, rram_throughput, width, label='RRAM', alpha=0.7)
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Throughput (Operations/Second)')
        ax.set_title('Throughput Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend()
        
        return fig
    
    def plot_stability_analysis(self,
                              benchmark_results: Dict[str, Any],
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot stability analysis.
        
        Args:
            benchmark_results: Results from stability benchmark
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        stability_metrics = benchmark_results.get('stability_metrics', {})
        
        if not stability_metrics:
            ax.text(0.5, 0.5, "No stability data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            return fig
        
        # Create a bar chart showing different stability metrics
        metrics = ['mean_deviation', 'std_deviation', 'max_deviation']
        values = [stability_metrics.get(m, 0) for m in metrics]
        
        ax.bar(metrics, values)
        ax.set_ylabel('Deviation Magnitude')
        ax.set_title('Stability Metrics Analysis')
        plt.xticks(rotation=45)
        
        return fig


def run_comprehensive_benchmark(
    rram_interface: Optional[object] = None,
    save_results: bool = True,
    output_dir: str = "comprehensive_benchmark"
) -> Dict[str, Any]:
    """
    Run a comprehensive benchmark of all available benchmarks.
    
    Args:
        rram_interface: Optional RRAM interface
        save_results: Whether to save results to files
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing all benchmark results
    """
    # Create benchmark suite
    benchmark_suite = RRAMBenchmarkSuite(
        rram_interface=rram_interface,
        save_results=save_results,
        results_dir=output_dir
    )
    
    all_results = {}
    
    print("Running Linear Solver Benchmark...")
    linear_results = benchmark_suite.run_linear_solver_benchmark(
        matrix_sizes=[10, 20, 50],
        num_samples=3
    )
    all_results['linear_solver'] = linear_results
    
    print("Running Power Efficiency Benchmark...")
    power_results = benchmark_suite.run_power_efficiency_benchmark(
        matrix_sizes=[10, 20, 50],
        duration=5.0
    )
    all_results['power_efficiency'] = power_results
    
    print("Running Throughput Benchmark...")
    throughput_results = benchmark_suite.run_throughput_benchmark(
        matrix_sizes=[10, 20, 50],
        duration=5.0
    )
    all_results['throughput'] = throughput_results
    
    print("Running Stability Benchmark...")
    stability_results = benchmark_suite.run_stability_benchmark(
        matrix_size=50,
        num_iterations=50
    )
    all_results['stability'] = stability_results
    
    # Generate report
    report = benchmark_suite.generate_benchmark_report()
    print("\nBenchmark Report:")
    print(report)
    
    if save_results:
        benchmark_suite.save_benchmark_results("comprehensive")
    
    return all_results