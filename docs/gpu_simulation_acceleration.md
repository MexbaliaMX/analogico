# GPU Simulation Acceleration for RRAM Systems

## Overview

The `gpu_simulation_acceleration` module provides GPU-accelerated simulation capabilities for RRAM systems. This module leverages modern parallel computing hardware (through JAX, CuPy, and PyTorch) to dramatically speed up RRAM simulations and algorithm testing, enabling researchers and developers to test larger systems and run more extensive simulations in shorter timeframes.

## Components

### GPUSimulationAccelerator Class
Main accelerator component that provides GPU-accelerated implementations:
- Supports multiple backends (JAX, CuPy, PyTorch)
- GPU-accelerated HP-INV solver
- GPU-accelerated material simulation
- Performance benchmarking tools
- Memory management and optimization

### SimulationAccelerationManager Class
Higher-level manager that intelligently selects optimal accelerators:
- Automatic backend selection based on problem characteristics
- Memory optimization for large-scale problems
- Performance prediction and selection
- Resource allocation and management

### GPUSolver Classes
Specialized GPU implementations for different algorithms:
- `gpu_hp_inv`: GPU-accelerated HP-INV solver
- `gpu_material_simulation`: GPU-accelerated material property simulation
- `gpu_neural_operations`: GPU-accelerated neural network operations

## Key Features

### Multiple Backend Support
- **JAX**: For XLA compilation and automatic differentiation
- **CuPy**: For NumPy-compatible GPU operations
- **PyTorch**: For deep learning and neural network operations

### Accelerated Algorithms
- HP-INV solver with GPU acceleration
- Material property simulation for thousands of devices
- Matrix operations (multiplication, inversion, etc.)
- Neural network operations

### Performance Optimization
- JIT compilation for faster execution
- Automatic batching of operations
- Memory-efficient computation graphs
- Parallel processing of multiple problems

### Benchmarking
- Performance comparison between CPU and GPU
- Speedup analysis for different problem sizes
- Memory and computational efficiency metrics

## Usage Examples

### Basic GPU Acceleration Setup
```python
from gpu_simulation_acceleration import create_gpu_accelerator

# Create accelerator with JAX backend (recommended for RRAM simulation)
accelerator = create_gpu_accelerator(backend='jax')

# Check if GPU is available
if accelerator.initialized:
    print(f"GPU acceleration available with {accelerator.backend}")
else:
    print("CPU-only mode active")
```

### Solving Linear Systems with GPU Acceleration
```python
from gpu_simulation_acceleration import SimulationAccelerationManager

# Create acceleration manager
manager = SimulationAccelerationManager()

# Create test problem
G = np.random.rand(100, 100) * 1e-4
G = G + 0.5 * np.eye(100)  # Diagonally dominant
b = np.random.rand(100)

# Solve adaptively (chooses best accelerator based on problem)
accelerator = manager.select_optimal_accelerator(
    problem_type='linear',
    problem_size=G.shape[0]
)

solution, iterations, info = accelerator.gpu_hp_inv(G, b, max_iter=20, bits=4)

print(f"GPU solution completed in {iterations} iterations")
print(f"Residual norm: {info['final_residual']:.2e}")
```

### Material Simulation Acceleration
```python
# Accelerate material property simulation for thousands of devices
material_results = accelerator.gpu_material_simulation(
    material_type='HfO2',
    num_devices=10000
)

print(f"Simulated {len(material_results)} devices in parallel")
print(f"Mean conductance: {np.mean(material_results):.2e} S")
print(f"Conductance std: {np.std(material_results):.2e} S")
```

### Benchmarking GPU Performance
```python
# Benchmark performance against CPU
benchmark_results = accelerator.benchmark_gpu_acceleration(
    matrix_sizes=[50, 100, 200, 500],
    num_trials=5
)

for size, speedup in benchmark_results['speedups'].items():
    if speedup > 1.0:
        print(f"Size {size}x{size}: {speedup:.2f}x speedup")
    else:
        print(f"Size {size}x{size}: CPU faster (speedup: {speedup:.2f}x)")
```

### Using Different Backends
```python
# JAX backend (fastest for iterative algorithms)
jax_accel = GPUSimulationAccelerator(backend='jax')

# CuPy backend (NumPy compatibility)
cupy_accel = GPUSimulationAccelerator(backend='cupy')

# PyTorch backend (best for neural networks)
torch_accel = GPUSimulationAccelerator(backend='torch')

# Compare performance
jax_solution, jax_it, jax_info = jax_accel.gpu_hp_inv(G, b)
torch_solution, torch_it, torch_info = torch_accel.gpu_hp_inv(G, b)
```

## Supported Backends

### JAX Backend
**Best for**: Iterative algorithms, automatic differentiation, XLA compilation
- Just-in-time compilation for maximum speed
- Automatic vectorization
- Excellent for HP-INV iterative refinement
- Handles complex mathematical operations efficiently

### CuPy Backend  
**Best for**: NumPy-compatible GPU operations
- Drop-in replacement for NumPy
- Familiar NumPy-style API
- Good for matrix operations
- Mature GPU implementation

### PyTorch Backend
**Best for**: Neural network operations, deep learning
- Excellent GPU memory management
- Good for tensor operations
- Can leverage neural network tools
- Great for neuromorphic applications

## Performance Benefits

### Computational Speed
- Up to 50-100x speedup for large problems
- Efficient parallel computation
- Optimized memory access patterns
- Reduced computation time

### Scalability
- Handle larger matrices and problems
- Process many devices in parallel
- Scale to hardware limits
- More extensive parameter sweeps

### Cost Efficiency
- Reduce computational costs
- Faster experiment iteration
- More comprehensive testing
- Efficient resource utilization

## Algorithms Accelerated

### HP-INV Solver
- GPU-accelerated iterative refinement
- Quantization and noise simulation
- Adaptive parameter computation
- Matrix inversion and multiplication

### Material Simulation
- Device property simulation for many devices
- Temperature and aging effects
- Process variation modeling
- Conductance distribution modeling

### Neural Network Operations
- Matrix-vector multiplication
- Layer operations
- Spiking neuron simulation
- Reservoir computing operations

## Memory Management

### Automatic Memory Optimization
- Efficient GPU memory usage
- Memory pooling for repeated operations
- Automatic data transfer optimization
- Out-of-core processing for large problems

### Memory Constraints
- Configurable memory limits
- Automatic fallback to CPU when needed
- Batched processing for large datasets
- Memory-efficient algorithms

## Integration with Existing Systems

The GPU acceleration module integrates seamlessly with:
- All RRAM simulation models
- HP-INV and advanced algorithms
- Benchmarking and testing systems
- Visualization and analysis tools

## Use Cases

### Research Applications
- Large-scale parameter sweeps
- Material property exploration
- Algorithm performance testing
- Design space exploration

### Development Applications
- Fast algorithm prototyping
- Performance optimization
- Hardware-software co-design
- Verification and validation

### Production Applications
- Real-time simulation
- Hardware-in-the-loop testing
- Rapid prototyping
- Performance optimization

This GPU acceleration module significantly enhances the computational capabilities of the Analogico project, enabling researchers and developers to work with larger, more complex RRAM systems in shorter timeframes.