# GPU Acceleration Testing for Analogico Project (Minimal Approach)

This document provides instructions for setting up and running the Analogico project with GPU acceleration using a minimalistic approach without Docker containers.

## Prerequisites

Before setting up GPU acceleration, ensure you have:

1. **NVIDIA GPU**: With compute capability >3.5
2. **NVIDIA GPU Drivers**: Properly installed with CUDA support
3. **Python**: Version 3.9-3.14 installed on your system
4. **System Dependencies**: Appropriate development tools for your OS

### Installing System Dependencies

For Fedora:
```bash
sudo dnf install python3-devel python3-pip gcc gcc-c++ openblas-devel lapack-devel gfortran
```

## Installing CUDA Toolkit (if needed)

If CUDA is not already installed on your system:

For Fedora:
```bash
# Add NVIDIA's repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo
sudo dnf install cuda-toolkit-12-4  # or appropriate version
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

### Option 2: Using Poetry (Alternative)

If you prefer using Poetry:

```bash
# Install project dependencies
poetry install

# Activate the environment
poetry shell

# Install JAX separately (since it's not in pyproject.toml for compatibility)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Verifying GPU Environment

### Basic Environment Check

```bash
# Activate your environment first
source analogico_env/bin/activate

# Verify basic dependencies
python -c "
import numpy as np
import scipy
import matplotlib
print('✓ NumPy version:', np.__version__)
print('✓ SciPy version:', scipy.__version__)
print('✓ Matplotlib version:', matplotlib.__version__)
"
```

### Testing GPU Availability

```bash
# Activate your environment
source analogico_env/bin/activate

# Check if JAX can see GPU devices
python -c "
try:
    import jax
    print(f'JAX version: {jax.__version__}')
    print(f'Available devices: {jax.devices()}')
    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
    print(f'GPU devices: {gpu_devices}')
    if gpu_devices:
        print('✓ GPU acceleration is available')
    else:
        print('⚠ No GPU devices found - CPU fallback will be used')
except ImportError as e:
    print(f'JAX not available: {e}')
    print('GPU acceleration will not be available, but CPU version works fine')
"
```

### Running Environment Validation

```bash
# Activate your environment
source analogico_env/bin/activate

# Run a comprehensive validation
python -c "
import numpy as np
print('NumPy version:', np.__version__)

try:
    import jax
    print('JAX version:', jax.__version__)
    print('JAX devices:', jax.devices())
    print('JAX backend:', jax.default_backend())
    
    # Test basic JAX operation
    import jax.numpy as jnp
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    z = jnp.dot(x, y)
    print(f'JAX dot product: {z}')
    print('✓ JAX is working correctly')
except Exception as e:
    print(f'JAX error: {e}')
    print('GPU acceleration unavailable, CPU fallback will be used')

try:
    import matplotlib
    print('Matplotlib version:', matplotlib.__version__)
    print('✓ Matplotlib is working correctly')
except Exception as e:
    print(f'Matplotlib error: {e}')

try:
    import scipy
    print('SciPy version:', scipy.__version__)
    print('✓ SciPy is working correctly')
except Exception as e:
    print(f'SciPy error: {e}')

try:
    import numba
    print('Numba version:', numba.__version__)
    print('✓ Numba is working correctly')
except Exception as e:
    print(f'Numba error: {e}')
"
```

## Running GPU Acceleration Tests

### Running the Conceptual GPU Test

```bash
# Activate your environment
source analogico_env/bin/activate

# Run the conceptual GPU test (validates infrastructure without requiring compatible runtime)
python gpu_end_to_end_test.py
```

### Testing GPU-Enabled Algorithms (if GPU is available)

```bash
# Activate your environment
source analogico_env/bin/activate

# Run a simple test to see if GPU acceleration works
python -c "
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), '.'))

try:
    from src.gpu_accelerated_hp_inv import GPUAcceleratedHPINV, gpu_available
    
    print(f'GPU acceleration available: {gpu_available()}')

    # Test the GPU solver
    if gpu_available():
        solver = GPUAcceleratedHPINV(use_gpu=True)
        print(f'✓ Using GPU: {solver.use_gpu}')
        
        # Create a test matrix
        n = 8
        G = np.random.rand(n, n) * 1e-4
        G = G + 0.5 * np.eye(n)  # Make well-conditioned
        b = np.random.rand(n)
        
        print(f'Testing GPU-accelerated HP-INV with {n}x{n} matrix...')
        
        # Solve using GPU acceleration
        solution, iterations, info = solver.solve(G, b, max_iter=10, bits=4)
        
        print(f'Solution found in {iterations} iterations')
        print(f'Final residual: {info[\"final_residual\"]:.2e}')
        print(f'Solution converged: {info[\"converged\"]}')
        
        # Verify the solution
        residual = np.linalg.norm(G @ solution - b)
        print(f'Verified residual: {residual:.2e}')
    else:
        print('GPU not available, will use CPU fallback')
        print('Algorithms will still work with reduced performance')
        
except Exception as e:
    print(f'Error during GPU test: {e}')
    import traceback
    traceback.print_exc()
"
```

## Performance Testing (Optional)

```bash
# Activate your environment
source analogico_env/bin/activate

# Run a simple performance comparison (CPU vs GPU if available)
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
   - To fix this, you may need to compile JAX from source:
     ```bash
     pip uninstall jax jaxlib
     pip install --upgrade \"jax[cuda]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
     ```
   - Or use CPU-only version: `pip install jax`

3. **GPU Not Detected**:
   - Ensure NVIDIA drivers and CUDA are properly installed
   - Verify with: `nvidia-smi`
   - Verify CUDA: `nvcc --version`

4. **Memory Issues**:
   - Set memory limits in JAX if needed:
     ```python
     import os
     os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'  # Use 50% of GPU memory
     ```

### Verification Commands

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check Python environment
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

## Notes

- The GPU acceleration in Analogico is designed with fallbacks - if GPU is not available, CPU implementations are used automatically
- All GPU functions have CPU equivalents for maximum compatibility
- For best performance, ensure your GPU has sufficient memory for your problem size
- The minimal approach allows for better debugging and system integration
- All functionality is preserved without Docker containers