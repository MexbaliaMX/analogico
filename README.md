# Analogico

Analogue-computing experiments around the HP-INV solver, including stochastic RRAM models, redundancy studies, Monte Carlo stress tests, and hardware-in-the-loop testing for high-precision inversion in resistive RAM (RRAM) systems.

## Repository Layout
- `src/`: HP-INV solver (`hp_inv.py`), RRAM generator (`rram_model.py`), redundancy pass, stress-test harness, hardware interface (`hardware_interface.py`), temperature compensation (`temperature_compensation.py`), and benchmarking (`benchmark.py`).
- `tests/`: pytest suites mirroring `src/`, including `tests/integration/` and hardware fixtures.
- `notebooks/`: Jupyter notebooks demonstrating algorithms with visualizations (`hp_inv_demonstration.ipynb`, `blockamc_scalability.ipynb`).
- `assets/`: reserved for datasets, plots, and hardware configs (add as experiments mature).
- `docs/`, `context.md`, `executive.md`, `Content.md`, `YouTube.md`, `List.md`: living research notes per AGENTS guidelines.
- `scripts/`: helper tooling and benchmarking scripts.
- `examples.py`: comprehensive usage examples.
- `APIDOC.md`: detailed API documentation.

## New Features
- **Enhanced HP-INV Algorithms**: Standard, block-based, adaptive, and recursive implementations with relaxation factors and convergence monitoring.
- **BlockAMC Scalability**: Implementation of BlockAMC algorithm for large matrix operations using smaller physical RRAM arrays.
- **Temperature Compensation**: Algorithms to compensate for temperature-induced drift in RRAM conductance values.
- **Hardware Interface**: Abstract interface and mock implementation for real RRAM device testing with realistic effects simulation.
- **Comprehensive Benchmarking**: Performance comparison against digital implementations with detailed metrics.
- **Jupyter Notebooks**: Interactive demonstrations with visualizations of all core algorithms.
- **Advanced RRAM Modeling**: Physics-based modeling including cycle-to-cycle variation, TDDB, conductive filament dynamics, material-specific characteristics, and ageing effects.
- **Performance Optimization**: Multi-core CPU acceleration, GPU acceleration with JAX, mixed-precision computing, sparse matrix support, and algorithm optimizations.
- **Material-Specific RRAM Models**: Detailed physics-based models for different RRAM materials (HfO₂, TaOₓ, TiO₂, NiOₓ, CoOₓ) with their specific switching mechanisms (VCM, ECM, mixed mechanisms).
- **GPU-Accelerated Simulation**: GPU-accelerated simulation and algorithm performance enhancement using JAX, CuPy, and PyTorch backends.
- **Real-time Adaptive Systems**: Real-time parameter adaptation based on changing environmental conditions, temperature, and device aging.
- **Advanced Visualization Tools**: Comprehensive visualization, dashboard, and data analysis tools for understanding RRAM behavior.
- **Comprehensive Testing Framework**: Complete testing ecosystem with unit, integration, stress, and property-based testing including:
  - Full pipeline integration tests that validate the complete workflow from RRAM matrix creation to HP-INV solution
  - Block HP-INV integration with RRAM models
  - Adaptive HP-INV algorithm validation
  - Stress testing framework with Monte Carlo simulations
  - RRAM model with redundancy and HP-INV integration tests
  - Hardware interface validation with mock implementations
  - Final validation tests that verify all newly implemented modules work together correctly
  - GPU acceleration and performance optimization validation
  - ML framework integration testing
  - Neuromorphic computing module validation
  - Real-time adaptive system testing
  - Enhanced visualization module validation

## Setup
1. Install Poetry if needed (`pip install poetry`).
2. Create/install the virtual environment:
   ```bash
   poetry install
   ```
   This reads `pyproject.toml`/`poetry.lock`, ensuring `python >=3.9,<3.14`.
3. Activate the shell (`poetry shell`) or prefix commands with `poetry run`.

## Running Examples
Run usage examples:
```bash
python examples.py
```

Run Jupyter notebooks:
```bash
jupyter notebook notebooks/
```

## Running Benchmarks
Run comprehensive benchmarks:
```bash
python src/benchmark.py
```

Or use the benchmark runner:
```bash
python benchmark_runner.py --type comprehensive
python benchmark_runner.py --type scalability --sizes 8 16 32
```

## Verification Workflow
Run these before commits/PRs to stay compliant with the AGENTS playbook:
1. **Unit & Integration Tests**
   ```bash
   poetry run pytest
   poetry run pytest tests/integration  # when integration suites are added
   # Or run integration tests directly:
   python tests/integration/test_full_pipeline.py
   ```
2. **Lint & Type Checks**
   ```bash
   poetry run invoke lint  # wraps ruff + mypy
   ```
3. **Benchmarks**
   ```bash
   poetry run invoke bench
   ```
   (Replace the placeholder script with full benchmarking/plotting as instrumentation is added.)
4. **End-to-End Validation**
   ```bash
   # Run comprehensive end-to-end validation
   python test_final_validation.py
   # Run all usage examples
   python examples.py
   ```

## Usage Examples

### Basic HP-INV Usage:
```python
import numpy as np
from src.hp_inv import hp_inv

# Create a test system
G = np.random.rand(4, 4) + 0.5 * np.eye(4)  # Well-conditioned matrix
b = np.random.rand(4)

# Solve using HP-INV
x, iters, info = hp_inv(G, b, bits=4, max_iter=20, tol=1e-8)

print(f"Solution found in {iters} iterations")
print(f"Final residual: {info['final_residual']:.2e}")
```

### RRAM Model with Temperature Effects:
```python
from src.rram_model import create_rram_matrix

# Create an RRAM matrix with temperature effects
G = create_rram_matrix(
    n=8,
    variability=0.03,              # 3% variability
    stuck_fault_prob=0.02,         # 2% stuck faults
    temperature=320,               # 47°C (320K)
    time_since_programming=3600    # 1 hour since programming
)
```

### Temperature Compensation:
```python
from src.temperature_compensation import apply_temperature_compensation_to_rram_matrix

# Compensate for temperature effects
G_compensated = apply_temperature_compensation_to_rram_matrix(
    G_original,
    temp_current=320,           # Current temperature in Kelvin
    temp_initial=300,           # Programming temperature
    time_elapsed=7200           # 2 hours since programming
)
```

## Contribution Checklist
- Maintain ≥90% coverage on critical solvers (`pytest --cov=src`).
- Document new modelling assumptions or experiments in the appropriate context files.
- Reference tickets in commit messages when available (`Refs #<id>`).
- Capture analogue run logs/plots when opening PRs.
