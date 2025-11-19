# Analogico: Analogue Computing Algorithms for High-Precision Inversion (HP-INV)

## Overview

Analogico is a Python project focused on analogue computing algorithms for High-Precision Inversion (HP-INV), designed to solve matrix equations using resistive RAM (RRAM) crossbars. The project implements iterative refinement algorithms that can achieve floating-point precision while maintaining the speed and efficiency benefits of analogue computing.

## Hardware Compatibility Note

**Important**: This project includes GPU acceleration using JAX, which may not be compatible with all CPU architectures due to AVX instruction requirements. The project has been updated to gracefully handle systems without AVX support:

- GPU-accelerated functions now include proper fallback to CPU implementations
- All core algorithms remain fully functional without GPU acceleration
- Performance is preserved through optimized CPU implementations

## Key Features

- **High-Precision Inversion (HP-INV)**: Enhanced iterative refinement algorithm with relaxation factors and convergence monitoring
- **Block HP-INV**: Scalability algorithm for large matrices using BlockAMC approach
- **RRAM Modeling**: Comprehensive RRAM models with discrete conductance levels, variability, stuck faults, and line resistance effects
- **GPU Acceleration**: Support for JAX, CuPy, and PyTorch backends (with CPU fallbacks)
- **Fault Tolerance**: Redundancy scheme for handling RRAM device failures
- **Temperature Compensation**: Algorithms to compensate for temperature-induced drift

## Installation

To install the project dependencies:

```bash
# For basic functionality (recommended)
pip install numpy scipy matplotlib jax jaxlib invoke

# For full functionality including performance optimizations:
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3-dev python3-pip build-essential libopenblas-dev liblapack-dev gfortran llvm-dev
pip install numpy scipy matplotlib jax jaxlib invoke numba

# On Fedora:
sudo dnf install python3-devel python3-pip gcc gcc-c++ openblas-devel lapack-devel gfortran llvm-devel
pip install numpy scipy matplotlib jax jaxlib invoke numba
```

Note: If you have compatibility issues with JAX, you can install the CPU-only version:
```bash
pip install jax[cpu]  # For systems with AVX support
# or
# Skip JAX installation entirely - all functions will use CPU fallbacks
```

## Optional Performance Dependencies

For enhanced performance with JIT compilation, install Numba:
- This may require system-level dependencies like LLVM
- The core functionality works without Numba, but with reduced performance
- To install: `pip install numba`

## Poetry Installation

If using Poetry for dependency management:
```bash
# Install system dependencies first (Ubuntu/Debian):
sudo apt install python3-dev python3-pip build-essential libopenblas-dev liblapack-dev gfortran llvm-dev

# Then install Python dependencies:
poetry install --with dev
```

## Usage

### Basic HP-INV Solution
```python
import numpy as np
from src.hp_inv import hp_inv

# Create a test matrix and vector
n = 10
G = np.random.rand(n, n) * 1e-4
G = G + 0.5 * np.eye(n)  # Ensure stability
b = np.random.rand(n)

# Solve using HP-INV
solution, iterations, info = hp_inv(G, b, max_iter=15, bits=4)
print(f'Solution converged in {iterations} iterations with residual {info["final_residual"]}')
```

### Using GPU-Accelerated Solver (with CPU fallback)
```python
from src.gpu_accelerated_hp_inv import GPUAcceleratedHPINV

# Initialize solver (will use CPU if GPU not available)
solver = GPUAcceleratedHPINV(use_gpu=True)

# Solve the system
solution, iterations, info = solver.solve(G, b, max_iter=15, bits=4)
print(f'Solution computed with {iterations} iterations')
```

## Architecture

### Core Components:

- `hp_inv.py`: Main HP-INV iterative refinement algorithm implementation
- `rram_model.py`: RRAM conductance matrix generation with discrete levels, variability, stuck faults, and line resistance effects
- `gpu_accelerated_hp_inv.py`: GPU implementations with CPU fallbacks
- `performance_optimization.py`: Multi-backend performance optimization with deferred imports

### Algorithms Implemented:

1. **HP-INV Solver**: Enhanced iterative refinement loop with relaxation factors and convergence monitoring
2. **Block HP-INV**: Scalability algorithm for large matrices using BlockAMC approach
3. **RRAM Model**: Stochastic RRAM models with device variability, stuck faults, line resistance, temperature effects and time-dependent drift
4. **GPU Acceleration**: Multi-backend support with JAX, CuPy, PyTorch and CPU fallbacks

## Performance

- 24-bit fixed-point accuracy (≈FP32) after ≤10 iterations
- 16×16 real-valued inversions using 8×8 arrays
- ~×10 throughput vs GPU/ASIC at N=128
- ×3–5 better energy efficiency compared to digital

## Research Context

This project implements the algorithms described in research on high-precision analogue matrix inversion using RRAM technology, addressing the precision bottleneck of classical analogue computing while maintaining its speed and energy efficiency advantages.

## License

This project is licensed under the MIT License.