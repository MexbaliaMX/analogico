# Project Overview: Analogico

## Description
Analogico is a Python project focused on analogue computing algorithms for High-Precision Inversion (HP-INV), designed to solve matrix equations using resistive RAM (RRAM) crossbars. The project implements iterative refinement algorithms that can achieve floating-point precision while maintaining the speed and efficiency benefits of analogue computing.

## Hardware Compatibility Note
**Important**: This project has been updated to support systems without AVX instruction support. GPU-accelerated functions now include proper CPU fallbacks, ensuring all core algorithms remain fully functional without GPU acceleration.

## Core Problem
Traditional digital matrix inversion has O(N³) complexity and is bottlenecked by memory bandwidth. Analogue matrix computing (AMC) uses RRAM crossbars as physical matrices to perform matrix-vector multiplication in O(1) time using Kirchhoff's laws. However, classical analogue circuits suffer from low precision (≈2-3 bits) due to device variability, noise, and coarse conductance levels.

## Solution: HP-INV Algorithm
The project implements a High-Precision Inversion scheme that combines:
- Low-precision inversion (LP-INV) with 3-bit quantization
- High-precision matrix-vector multiplication (HP-MVM) using bit-slicing
- Iterative refinement entirely in the analogue domain

The algorithm decomposes matrix A into 3-bit slices (A = Σ 2^{im} A_i), with the most significant slice (A₀) mapped to LP-INV and remaining slices driving HP-MVM via bit-slicing.

## Key Components

### Source Files
- `hp_inv.py`: Main HP-INV iterative refinement algorithm implementation
- `rram_model.py`: RRAM conductance matrix generation with discrete levels, variability, stuck faults, and line resistance effects
- `redundancy.py`: Redundancy scheme for fault tolerance using row/column substitution
- `stress_test.py`: Monte Carlo stress tests for evaluating HP-INV under RRAM variability
- `gpu_accelerated_hp_inv.py`: GPU implementations with CPU fallbacks for systems without AVX support

### Key Features
- Support for 8 stable conductance levels (0.5–35 μS) with ASAP write-verify algorithm
- Conductance variability modeling and stuck-at fault simulation
- Line resistance effects modeling
- Iterative refinement with configurable precision and noise parameters
- Multi-backend GPU acceleration (JAX, CuPy, PyTorch) with CPU fallbacks
- Deferred JAX imports to handle AVX compatibility issues

## Technical Stack
- Python >=3.9, <3.14
- NumPy for numerical computations
- JAX for advanced numerical computing (with CPU-only fallback for non-AVX systems)
- Matplotlib for plotting
- SciPy for scientific computing
- Invoke for task automation

## Hardware Compatibility Improvements

### GPU Acceleration with CPU Fallbacks
The project has been enhanced with a robust fallback system:
- All GPU-accelerated functions defer JAX imports until function execution
- CPU fallback implementations are provided for all GPU-dependent functions
- Detection of hardware capabilities happens at runtime
- Performance is preserved through optimized CPU implementations

### Deferred Import Strategy
- JAX imports are now handled at function level rather than module level
- This prevents AVX-related errors during import on incompatible systems
- Functions check for JAX availability when called, not when defined
- Import errors are caught and handled gracefully with CPU alternatives

### Multi-backend Support
- Support for JAX, CuPy, and PyTorch with automatic fallback to NumPy
- Runtime detection of available backends
- Consistent API across all backend implementations
- Automatic selection of best available computational backend

## Project Structure
- `src/`: Core analogue computing algorithms
- `tests/`: Pytest suites with unit and integration tests
- `notebooks/`: Exploratory analysis notebooks
- `docs/`: Documentation files
- `scripts/`: Helper tooling

## Algorithms Implemented
1. **HP-INV Solver**: Enhanced iterative refinement loop with relaxation factors and convergence monitoring
2. **Block HP-INV**: Scalability algorithm for large matrices using BlockAMC approach
3. **BlockAMC Inversion**: Full implementation of BlockAMC algorithm with recursive block diagonal inversion
4. **Recursive Block Inversion**: Advanced recursive implementation of block matrix inversion
5. **Adaptive HP-INV**: Dynamic precision adjustment based on convergence behavior
6. **RRAM Model**: Stochastic RRAM models with device variability, stuck faults, line resistance, temperature effects and time-dependent drift
7. **Advanced RRAM Models**: Physics-based modeling including cycle-to-cycle variation, TDDB, conductive filament dynamics, material-specific characteristics, and ageing effects
8. **Material-Specific RRAM Models**: Detailed physics-based models for HfO₂, TaOₓ, TiO₂, NiOₓ, and CoOₓ materials with their specific switching mechanisms (VCM, ECM, mixed mechanisms)
9. **Performance Optimization**: Multi-core CPU acceleration, GPU acceleration with JAX, mixed-precision computing, sparse matrix support, and algorithm optimizations
10. **Redundancy**: Fault tolerance through row/column substitution
11. **Stress Tests**: Monte Carlo tests for reliability evaluation with temperature and time effects
12. **Hardware Interface**: Abstract interface and mock implementation for real RRAM device testing
13. **Temperature Compensation**: Algorithms to compensate for temperature-induced drift in RRAM conductance values

## Testing Approach
- Unit tests using pytest for individual components
- Integration tests for system-level validation
- Property-based testing with hypothesis
- Monte Carlo stress tests for reliability evaluation
- Target ≥90% coverage on critical solvers

## Hardware Platform
- Foundry-fabricated 40nm TaOₓ 1T1R RRAM arrays
- 8×8 arrays for LP-INV, 1 Mb array for HP-MVM
- Operational amplifiers (AD823, OPA690) for inversion loops
- ~120ns LP-INV transient response, ~60ns LP-MVM

## Performance Results
- 24-bit fixed-point accuracy (≈FP32) after ≤10 iterations
- 16×16 real-valued inversions using 8×8 arrays
- ~×10 throughput vs GPU/ASIC at N=128
- ×3–5 better energy efficiency compared to digital

## Applications
- Massive MIMO zero-forcing detection with 256-QAM modulation
- Linear system solving for scientific computing
- RF signal processing
- Second-order ML training

## Testing and Validation

### End-to-End Testing
The project has been thoroughly tested with the following procedures:
- GPU acceleration functionality tested with CPU fallback mechanisms
- Core algorithms validated to work on systems without AVX support
- All HP-INV implementations (standard, block, adaptive) tested for functionality
- Performance benchmarks run to verify CPU fallback efficiency
- Import functionality tested to ensure no AVX-related errors on startup

### Hardware Compatibility Testing
- Tested import of all modules without triggering JAX AVX errors
- Verified CPU fallbacks work correctly when GPU acceleration unavailable
- Confirmed all core algorithms maintain precision on CPU-only systems
- Validated that GPU-accelerated functions can be called safely with fallbacks

## Development Guidelines
- Follows PEP 8 with 4-space indentation
- Type hints required throughout
- Use `poetry install` for setup
- `poetry run pytest` for testing (with CPU-only fallback tests)
- `poetry run invoke lint` for code quality checks

## Research Context
This project implements the algorithms described in research on high-precision analogue matrix inversion using RRAM technology, addressing the precision bottleneck of classical analogue computing while maintaining its speed and energy efficiency advantages.