# GPU Acceleration Testing for Analogico Project (Minimalistic Approach)

This document provides instructions for setting up and running the Analogico project with GPU acceleration using a minimalistic approach without Docker containers.

## Prerequisites

Before setting up GPU acceleration, ensure you have:

1. **NVIDIA GPU**: With compute capability >3.5
2. **NVIDIA GPU Drivers**: Properly installed with CUDA support
3. **Python**: Version 3.9-3.14 installed on your system
4. **System Dependencies**: Appropriate development tools for your OS

### Installing System Dependencies

For performance optimizations with Numba JIT compilation, additional system dependencies are required:

For Fedora:
```bash
sudo dnf install python3-devel python3-pip gcc gcc-c++ openblas-devel lapack-devel gfortran llvm-devel
```

For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3-dev python3-pip build-essential libopenblas-dev liblapack-dev gfortran llvm-dev
```

## Installing CUDA Toolkit (if needed)

If CUDA is not already installed on your system:

For Fedora:
```bash
# Add NVIDIA's repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo
sudo dnf install cuda-toolkit-12-4  # or appropriate version
```

For Ubuntu/Debian:
```bash
# Download and install CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-4
```

After installation, add CUDA paths to your environment:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Setting up the Environment

### Option 1: Using Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv analogico_env

# Activate the environment
source analogico_env/bin/activate  # On Windows: analogico_env\Scripts\activate

# Upgrade pip to the latest version
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install numpy scipy matplotlib invoke

# Install JAX with CUDA support
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install remaining performance dependencies
pip install jaxlib numba

# Install development dependencies
pip install pytest hypothesis
```

## Running GPU Acceleration Tests

### Running the End-to-End GPU Test

```bash
# Activate your environment
source analogico_env/bin/activate

# Run the conceptual GPU test (validates infrastructure without requiring compatible runtime)
python gpu_end_to_end_test.py
```

### Running an Interactive Session

```bash
# Activate your environment
source analogico_env/bin/activate

# Start Python
python
```

Once in Python, you can run:

```python
import sys
sys.path.insert(0, '.')
from src.gpu_accelerated_hp_inv import GPUAcceleratedHPINV, gpu_available
print(f'GPU acceleration available: {gpu_available()}')

# Test the GPU solver
if gpu_available():
    solver = GPUAcceleratedHPINV(use_gpu=True)
    print(f'Using GPU: {solver.use_gpu}')
else:
    print('GPU not available, will use CPU fallback')
```

## Testing Full HP-INV Functionality

In your activated environment, run:

```bash
python -c "
import numpy as np
import sys
sys.path.insert(0, '.')

# Test the HP-INV functionality with GPU acceleration
from src.gpu_accelerated_hp_inv import GPUAcceleratedHPINV

# Create a test matrix
n = 8
G = np.random.rand(n, n) * 1e-4
G = G + 0.5 * np.eye(n)  # Make well-conditioned
b = np.random.rand(n)

print(f'Testing GPU-accelerated HP-INV with {n}x{n} matrix...')

# Initialize the GPU solver
gpu_solver = GPUAcceleratedHPINV(use_gpu=True)
print(f'Using GPU: {gpu_solver.use_gpu}')

# Solve using GPU acceleration
try:
    import time
    start_time = time.time()
    solution, iterations, info = gpu_solver.solve(G, b, max_iter=15, bits=4)
    end_time = time.time()

    print(f'Solution found in {iterations} iterations')
    print(f'Time taken: {end_time - start_time:.4f} seconds')
    print(f'Final residual: {info[\"final_residual\"]:.2e}')
    print(f'Solution converged: {info[\"converged\"]}')

    # Verify the solution
    residual = np.linalg.norm(G @ solution - b)
    print(f'Verified residual: {residual:.2e}')

except Exception as e:
    print(f'Error during GPU HP-INV: {e}')
    import traceback
    traceback.print_exc()
"
```

## Troubleshooting

### Common Issues

1. **JAX Installation Fails**:
   - Ensure CUDA is properly installed and paths are set
   - Try installing specific CUDA-compatible JAX version:
     ```bash
     pip install 'jax[cuda12_pip] @ https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
     ```

2. **AVX Instruction Issues**:
   - On some systems, pre-compiled JAX wheels may have AVX compatibility issues
   - To fix this, you may need to compile JAX from source or use CPU-only version: `pip install jax`

3. **GPU Not Detected**:
   - Ensure NVIDIA drivers and CUDA are properly installed
   - Verify with: `nvidia-smi`
   - Verify CUDA: `nvcc --version`

4. **Python Version Issues**:
   - The project works with Python 3.9-3.14, ensure you're using a supported version

### Verification Commands

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Activate your environment and check Python environment
source analogico_env/bin/activate
python -c "import jax; print(jax.devices())"

# Validate all dependencies
python -c "
import numpy as np
import scipy
import matplotlib
print('Core packages imported successfully')

try:
    import jax
    print(f'JAX available: {jax.__version__}')
    print(f'JAX devices: {jax.devices()}')
except ImportError:
    print('JAX not available - GPU acceleration will not work')
"
```

## Performance Testing

```bash
# Activate your environment
source analogico_env/bin/activate

# Run a performance comparison test
python -c "
import numpy as np
import sys
import os
import time
sys.path.insert(0, os.path.join(os.getcwd(), '.'))

try:
    from src.gpu_accelerated_hp_inv import GPUAcceleratedHPINV
    from src.hp_inv import hp_inv  # CPU implementation

    # Create test matrices of different sizes
    sizes = [16, 32, 64]

    for n in sizes:
        print(f'\\nTesting with {n}x{n} matrix:')

        # Create a well-conditioned test matrix
        G = np.random.rand(n, n) * 1e-4
        G = G + 0.5 * np.eye(n)
        b = np.random.rand(n)

        # Test CPU performance
        start_time = time.time()
        cpu_solution, cpu_iters, cpu_info = hp_inv(G, b, max_iter=10, bits=4)
        cpu_time = time.time() - start_time

        print(f'  CPU time: {cpu_time:.4f}s, {cpu_iters} iterations')

        # Test GPU performance if available
        gpu_solver = GPUAcceleratedHPINV(use_gpu=True)
        if gpu_solver.use_gpu:
            start_time = time.time()
            gpu_solution, gpu_iters, gpu_info = gpu_solver.solve(G, b, max_iter=10, bits=4)
            gpu_time = time.time() - start_time

            print(f'  GPU time: {gpu_time:.4f}s, {gpu_iters} iterations')
            if cpu_time > 0 and gpu_time > 0:
                print(f'  Speedup: {cpu_time/gpu_time:.2f}x')
        else:
            print('  GPU not available, using CPU fallback')
    print('\\n✓ Performance test completed')
except Exception as e:
    print(f'Performance test error: {e}')
"
```

## Notes

- The GPU acceleration in Analogico is designed with fallbacks - if GPU is not available, CPU implementations are used automatically
- All GPU functions have CPU equivalents for maximum compatibility
- For best performance, ensure your GPU has sufficient memory for your problem size
- The minimal approach allows for better debugging and system integration
- All functionality is preserved without Docker containers