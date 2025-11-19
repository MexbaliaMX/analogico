# Analogico Project API Documentation

## Overview
The Analogico project implements high-precision inversion (HP-INV) algorithms for RRAM-based analogue computing. This document provides API references and usage examples for the key modules.

## Installation

```bash
# Using poetry (recommended)
poetry install

# Or directly with pip
pip install numpy jax matplotlib scipy invoke
```

## Core Modules

### 1. HP-INV Algorithms (`src.hp_inv`)

#### `hp_inv(G, b, bits=3, lp_noise_std=0.01, max_iter=10, tol=1e-6, relaxation_factor=1.0)`

Solve linear system `Gx = b` using the HP-INV iterative refinement algorithm.

**Parameters:**
- `G: np.ndarray` - Conductance matrix (with variability)
- `b: np.ndarray` - Right-hand side vector
- `bits: int` - Bit precision for LP-INV quantization (default: 3)
- `lp_noise_std: float` - Standard deviation of noise added to LP-INV (default: 0.01)
- `max_iter: int` - Maximum iterations (default: 10)
- `tol: float` - Tolerance for convergence (default: 1e-6)
- `relaxation_factor: float` - Relaxation factor for convergence acceleration (default: 1.0)

**Returns:**
- `Tuple[np.ndarray, int, dict]` - (solution x, iterations taken, convergence info)

**Example:**
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
print(f"Converged: {info['converged']}")
```

#### `block_hp_inv(G, b, block_size=4, **kwargs)`

Solve large linear systems using BlockAMC algorithm with block-partitioning approach.

**Parameters:**
- `G: np.ndarray` - Large conductance matrix to invert
- `b: np.ndarray` - Right-hand side vector
- `block_size: int` - Size of submatrix blocks (default: 4)
- `**kwargs` - Additional arguments passed to `hp_inv`

**Returns:**
- `Tuple[np.ndarray, int, dict]` - (solution x, iterations taken, convergence info)

**Example:**
```python
import numpy as np
from src.hp_inv import block_hp_inv
from src.rram_model import create_rram_matrix

# Create a large RRAM matrix
G_large = create_rram_matrix(16, variability=0.03, stuck_fault_prob=0.01)
G_large = G_large + 0.3 * np.eye(16)  # Ensure well-conditioned
b_large = np.random.rand(16)

# Solve using Block HP-INV
x_block, iters_block, info_block = block_hp_inv(G_large, b_large, block_size=8)
print(f"Block HP-INV completed in {iters_block} iterations")
```

#### `adaptive_hp_inv(G, b, initial_bits=3, max_bits=6, **kwargs)`

Solve using adaptive precision that adjusts based on convergence behavior.

**Parameters:**
- `G: np.ndarray` - Conductance matrix
- `b: np.ndarray` - Right-hand side vector
- `initial_bits: int` - Initial quantization bits (default: 3)
- `max_bits: int` - Maximum allowed bits (default: 6)
- `**kwargs` - Additional arguments passed to `hp_inv`

**Returns:**
- `Tuple[np.ndarray, int, dict]` - (solution x, iterations taken, convergence info)

**Example:**
```python
import numpy as np
from src.hp_inv import adaptive_hp_inv

G = np.random.rand(6, 6) + 0.5 * np.eye(6)
b = np.random.rand(6)

x_adaptive, iters, info = adaptive_hp_inv(G, b, initial_bits=2, max_bits=5, max_iter=15)
print(f"Adaptive HP-INV used {info['final_bits']} bits and {iters} iterations")
```

#### `blockamc_inversion(G, block_size=8)`

Full implementation of BlockAMC algorithm for matrix inversion.

**Parameters:**
- `G: np.ndarray` - Matrix to invert
- `block_size: int` - Size of physical RRAM tiles (default: 8)

**Returns:**
- `np.ndarray` - Inverted matrix

**Example:**
```python
import numpy as np
from src.hp_inv import blockamc_inversion

G = np.random.rand(16, 16) + 0.5 * np.eye(16)
G_inv = blockamc_inversion(G, block_size=8)

# Verify the inverse
identity_result = G @ G_inv
error = np.linalg.norm(identity_result - np.eye(16))
print(f"Inversion error: {error:.2e}")
```

#### `recursive_block_inversion(G, block_size=8, depth=0, max_depth=3)`

Recursive implementation of block matrix inversion based on BlockAMC principles.

**Parameters:**
- `G: np.ndarray` - Matrix to invert
- `block_size: int` - Base tile size for physical RRAM arrays (default: 8)
- `depth: int` - Current recursion depth (default: 0)
- `max_depth: int` - Maximum recursion depth (default: 3)

**Returns:**
- `np.ndarray` - Inverted matrix

### 2. RRAM Model (`src.rram_model`)

#### `create_rram_matrix(n, conductance_levels=DEFAULT_CONDUCTANCE_LEVELS, variability=0.05, stuck_fault_prob=0.01, line_resistance=1.7e-3, temperature=300.0, time_since_programming=0.0, temp_coeff=0.002, activation_energy=0.5, use_advanced_physics=False, material='HfO2', device_area=1e-12)`

Create an n×n RRAM conductance matrix with realistic properties.

**Parameters:**
- `n: int` - Matrix size
- `conductance_levels: List[float]` - Discrete conductance levels in Siemens (default: 8 levels from 0.5µS to 35µS)
- `variability: float` - Relative standard deviation for conductance deviations (default: 0.05)
- `stuck_fault_prob: float` - Probability of stuck-at fault per cell (default: 0.01)
- `line_resistance: float` - Wire resistance per adjacent device in Ω (default: 1.7e-3)
- `temperature: float` - Operating temperature in Kelvin (default: 300.0)
- `time_since_programming: float` - Time elapsed since programming in seconds (default: 0.0)
- `temp_coeff: float` - Temperature coefficient for compensation (default: 0.002)
- `activation_energy: float` - Activation energy for time-dependent drift (default: 0.5 eV)
- `use_advanced_physics: bool` - Whether to use advanced physics-based models (default: False)
- `material: str` - RRAM material type ('HfO2', 'TaOx', 'TiO2', etc.) (default: 'HfO2')
- `device_area: float` - Physical area of each device in m² (default: 1e-12)

**Returns:**
- `np.ndarray` - Conductance matrix G with RRAM properties

**Example:**
```python
from src.rram_model import create_rram_matrix

# Create an 8x8 RRAM matrix with realistic parameters
G = create_rram_matrix(
    n=8,
    variability=0.03,        # 3% variability
    stuck_fault_prob=0.005,  # 0.5% stuck faults
    line_resistance=1.5e-3   # 1.5 mΩ line resistance
)
print(f"Matrix shape: {G.shape}")
print(f"Number of stuck faults: {len(G[G == 0])}")
print(f"Conductance range: {G.min():.2e} - {G.max():.2e} S")
```

#### `mvm(G, x)`

Perform matrix-vector multiplication with conductance matrix.

**Parameters:**
- `G: np.ndarray` - Conductance matrix
- `x: np.ndarray` - Input vector

**Returns:**
- `np.ndarray` - Output vector y = G @ x

**Example:**
```python
from src.rram_model import create_rram_matrix, mvm

G = create_rram_matrix(4, variability=0.02)
x = np.ones(4)

y = mvm(G, x)
print(f"Result of MVM: {y}")
```

### 3. Redundancy (`src.redundancy`)

#### `apply_redundancy(G)`

Apply redundancy scheme to detect and repair stuck-at faults.

**Parameters:**
- `G: np.ndarray` - Conductance matrix with potential stuck faults

**Returns:**
- `np.ndarray` - Repaired matrix

**Example:**
```python
from src.rram_model import create_rram_matrix
from src.redundancy import apply_redundancy

# Create matrix with stuck faults
G_with_faults = create_rram_matrix(8, stuck_fault_prob=0.05)
print(f"Before repair: {len(G_with_faults[G_with_faults == 0])} stuck faults")

# Apply redundancy
G_repaired = apply_redundancy(G_with_faults)
print(f"After repair: {len(G_repaired[G_repaired == 0])} stuck faults remaining")
```

### 4. Stress Testing (`src.stress_test`)

#### `run_stress_test(n=8, num_samples=100, variability=0.05, stuck_prob=0.01, bits=3, lp_noise_std=0.01, max_iter=10)`

Run Monte Carlo stress test for HP-INV under RRAM variability.

**Parameters:**
- `n: int` - Matrix size (default: 8)
- `num_samples: int` - Number of Monte Carlo samples (default: 100)
- `variability: float` - Conductance variability (default: 0.05)
- `stuck_prob: float` - Stuck-at fault probability (default: 0.01)
- `bits: int` - LP-INV bit precision (default: 3)
- `lp_noise_std: float` - LP-INV noise std (default: 0.01)
- `max_iter: int` - Max iterations for HP-INV (default: 10)

**Returns:**
- `Tuple[list[int], list[float]]` - (convergence_iterations, final_relative_errors)

**Example:**
```python
from src.stress_test import run_stress_test

# Run a small stress test
iters, errors = run_stress_test(
    n=6, 
    num_samples=50, 
    variability=0.03, 
    stuck_prob=0.02,
    bits=4
)

print(f"Average iterations: {sum(iters)/len(iters):.2f}")
print(f"Average error: {sum(errors)/len(errors):.2e}")
print(f"Success rate: {len(iters)/50*100:.1f}%")
```

### 5. Hardware Interface (`src.hardware_interface`)

#### `MockRRAMHardware(size=8, variability=0.05, stuck_fault_prob=0.01, line_resistance=1.7e-3)`

Mock RRAM hardware implementation for testing with realistic effects.

**Parameters:**
- `size: int` - Size of the square RRAM crossbar array (default: 8)
- `variability: float` - Conductance variability (default: 0.05)
- `stuck_fault_prob: float` - Stuck-at fault probability (default: 0.01)
- `line_resistance: float` - Line resistance effects (default: 1.7e-3)

**Methods:**
- `connect()` - Connect to the hardware device
- `disconnect()` - Disconnect from the hardware device
- `configure(**config)` - Configure the hardware device
- `write_matrix(matrix)` - Write a conductance matrix to the RRAM crossbar
- `read_matrix()` - Read the current conductance matrix
- `matrix_vector_multiply(vector)` - Perform MVM operation
- `invert_matrix(matrix)` - Perform matrix inversion operation

**Example:**
```python
from src.hardware_interface import MockRRAMHardware
import numpy as np

# Create mock hardware
hardware = MockRRAMHardware(size=4, variability=0.02, stuck_fault_prob=0.01)

# Connect and program a matrix
hardware.connect()
test_matrix = np.random.rand(4, 4) * 1e-6
test_matrix = test_matrix + 0.5 * np.eye(4)

success = hardware.write_matrix(test_matrix)
if success:
    print("Matrix successfully programmed to hardware")

    # Perform MVM
    vector = np.ones(4)
    result = hardware.matrix_vector_multiply(vector)
    print(f"MVM result: {result}")

    # Perform inversion
    inv_result = hardware.invert_matrix(test_matrix)
    print(f"Inversion completed")

hardware.disconnect()
```

### 6. Arduino RRAM Interface (`src.arduino_rram_interface`)

#### `ArduinoRRAMInterface(port='/dev/ttyUSB0', protocol='serial', **config)`

Arduino-based RRAM interface for simulating analog matrix operations.
Communicates via configurable protocol to an Arduino configured as an RRAM emulator.

**Parameters:**
- `port: str` - Serial port connected to Arduino (default: '/dev/ttyUSB0')
- `protocol: str` - Communication protocol ('serial', 'spi', 'i2c') (default: 'serial')
- `**config` - Additional configuration parameters

**Methods:**
- `connect()` - Connect to the Arduino device via configured protocol
- `disconnect()` - Disconnect from the Arduino device
- `configure(**config)` - Configure the Arduino RRAM device
- `write_matrix(matrix)` - Write a conductance matrix to the Arduino RRAM crossbar
- `read_matrix()` - Read the current conductance matrix from Arduino
- `matrix_vector_multiply(vector)` - Perform MVM operation on Arduino
- `invert_matrix(matrix)` - Perform matrix inversion on Arduino

**Example:**
```python
from src.arduino_rram_interface import ArduinoRRAMInterface
import numpy as np

# Create Arduino interface
arduino = ArduinoRRAMInterface(port='/dev/ttyUSB0')

try:
    # Connect to Arduino
    arduino.connect()
    print("Connected to Arduino RRAM device")

    # Create test matrix
    test_matrix = np.random.rand(4, 4) * 1e-6
    test_matrix = test_matrix + 0.5 * np.eye(4)

    # Program matrix to Arduino
    success = arduino.write_matrix(test_matrix)
    if success:
        print("Matrix successfully programmed to Arduino")

        # Perform MVM
        vector = np.ones(4)
        result = arduino.matrix_vector_multiply(vector)
        print(f"MVM result: {result}")

        # Perform inversion
        inv_result = arduino.invert_matrix(test_matrix)
        print(f"Inversion completed")

    arduino.disconnect()
except RuntimeError as e:
    print(f"Hardware error: {e}")
```

#### `MultiArduinoRRAMInterface(ports, **base_params)`

Interface for managing multiple Arduino RRAM devices working in parallel.

**Parameters:**
- `ports: List[str]` - List of serial ports for each Arduino
- `**base_params` - Parameters to pass to each Arduino interface

**Methods:**
- `connect_all()` - Connect to all Arduino devices
- `disconnect_all()` - Disconnect from all Arduino devices
- `distribute_matrix(matrix, block_size)` - Distribute a large matrix across multiple Arduino devices
- `block_matrix_multiply(A, B, block_size)` - Perform block matrix multiplication using multiple Arduinos
- `block_matrix_vector_multiply(matrix, vector, block_size)` - Perform block MVM using multiple Arduinos
- `block_matrix_inversion(matrix, block_size)` - Perform block matrix inversion using multiple Arduinos

#### `ArduinoRRAMDemo(interface, use_gpu=False)`

A demonstration class to showcase how to use the Arduino RRAM interface
with HP-INV solver for solving linear systems.

**Parameters:**
- `interface: ArduinoRRAMInterface` - The Arduino interface to use
- `use_gpu: bool` - Whether to use GPU acceleration for computation (default: False)

**Methods:**
- `demonstrate_mvm(test_matrix, test_vector)` - Demonstrate matrix-vector multiplication
- `demonstrate_inversion(test_matrix)` - Demonstrate matrix inversion using Arduino
- `demonstrate_hp_inv_with_arduino(G, b)` - Demonstrate HP-INV using Arduino for some operations

#### `BlockAMCSolver(multi_arduino_interface)`

BlockAMC solver using multiple Arduino RRAM devices for large matrix operations.

**Parameters:**
- `multi_arduino_interface: MultiArduinoRRAMInterface` - Interface managing multiple Arduino devices

**Methods:**
- `solve(G, b, block_size=8)` - Solve the linear system G*x = b using BlockAMC with multiple Arduinos
- `invert_large_matrix(matrix, block_size=8)` - Invert a large matrix using BlockAMC with multiple Arduino tiles

### 7. Advanced RRAM Models (`src.advanced_rram_models`)

#### `AdvancedRRAMModel(material=RRAMMaterialType.HFO2, device_area=1e-12, temperature=300.0, initial_resistance_state="high")`

Advanced physics-based RRAM device model with multiple physical effects.

**Parameters:**
- `material: RRAMMaterialType` - Type of RRAM material (default: RRAMMaterialType.HFO2)
- `device_area: float` - Physical area of the device in m² (default: 1e-12)
- `temperature: float` - Operating temperature in Kelvin (default: 300.0)
- `initial_resistance_state: str` - Initial state "high" or "low" (default: "high")

**Methods:**
- `apply_voltage(voltage, duration)` - Apply voltage and simulate device response
- `get_conductance()` - Get current conductance
- `get_device_state()` - Get comprehensive device state
- `check_breakdown()` - Check for dielectric breakdown

**Example:**
```python
from src.advanced_rram_models import AdvancedRRAMModel, RRAMMaterialType

# Create an advanced HfO2 device model
device = AdvancedRRAMModel(
    material=RRAMMaterialType.TAOX,
    device_area=1e-12,
    temperature=320  # Higher temperature
)

# Apply a SET voltage pulse
result = device.apply_voltage(-2.0, 1e-8)  # -2V, 10ns pulse
print(f"Switched: {result['switched']}, New resistance: {result['resistance_after']:.2e}Ω")
```

#### `DeviceArrayModel(rows, cols, material=RRAMMaterialType.HFO2)`

Model for arrays of RRAM devices with crossbar parasitics and interactions.

**Parameters:**
- `rows: int` - Number of rows in the crossbar
- `cols: int` - Number of columns in the crossbar
- `material: RRAMMaterialType` - Material type for all devices (default: RRAMMaterialType.HFO2)

**Methods:**
- `apply_voltage_vector(row_voltages, col_voltages)` - Apply voltages to crossbar
- `update_device_states(row_voltages, col_voltages, duration)` - Update all device states

**Example:**
```python
from src.advanced_rram_models import DeviceArrayModel

# Create a 4x4 crossbar array
array = DeviceArrayModel(4, 4, material=RRAMMaterialType.HFO2)

# Apply voltages to select a device
row_voltages = [-2.0, 0, 0, 0]  # Activate first row
col_voltages = [0, 0, 0, 0]     # Ground all columns

currents, voltages = array.apply_voltage_vector(row_voltages, col_voltages)
print(f"Current through selected device: {currents[0,0]:.2e}A")
```

### 8. Performance Optimization (`src.performance_optimization`)

#### `PerformanceOptimizer(use_jax=True, use_numba=True, n_cores=None)`

Main class for performance optimization of RRAM algorithms.

**Parameters:**
- `use_jax: bool` - Whether to use JAX for GPU acceleration (default: True)
- `use_numba: bool` - Whether to use Numba for CPU JIT compilation (default: True)
- `n_cores: int` - Number of CPU cores to use (default: all available)

**Methods:**
- `optimize_rram_model()` - Returns optimized quantization and MVM functions

**Example:**
```python
from src.performance_optimization import PerformanceOptimizer

optimizer = PerformanceOptimizer()
quantize_func, mvm_func = optimizer.optimize_rram_model()
```

#### `parallel_stress_test(n, num_samples, variability=0.05, stuck_prob=0.01, bits=3, lp_noise_std=0.01, max_iter=10, n_processes=None)`

Parallel stress test using multiprocessing for better performance.

**Parameters:**
- `n: int` - Matrix size
- `num_samples: int` - Number of Monte Carlo samples
- `variability: float` - Conductance variability
- `stuck_prob: float` - Stuck-at fault probability
- `bits: int` - LP-INV bit precision
- `lp_noise_std: float` - LP-INV noise std
- `max_iter: int` - Max iterations for HP-INV
- `n_processes: int` - Number of processes to use (default: all available)

**Returns:**
- `Tuple[list[int], list[float]]` - (convergence_iterations, final_relative_errors)

**Example:**
```python
from src.performance_optimization import parallel_stress_test

iters, errors = parallel_stress_test(
    n=8, 
    num_samples=100, 
    n_processes=4  # Use 4 CPU cores
)
print(f"Completed {len(iters)} stress tests in parallel")
```

#### `jax_hp_inv(G, b, bits=3, lp_noise_std=0.01, max_iter=10, tol=1e-6, relaxation_factor=1.0)`

JAX-optimized HP-INV algorithm for GPU acceleration.

**Parameters:**
- `G: np.ndarray` - Conductance matrix (with variability)
- `b: np.ndarray` - Right-hand side vector
- `bits: int` - Bit precision for LP-INV quantization (default: 3)
- `lp_noise_std: float` - Standard deviation of noise added to LP-INV (default: 0.01)
- `max_iter: int` - Maximum iterations (default: 10)
- `tol: float` - Tolerance for convergence (default: 1e-6)
- `relaxation_factor: float` - Relaxation factor for convergence acceleration (default: 1.0)

**Returns:**
- `Tuple[np.ndarray, int, dict]` - (solution x, iterations taken, convergence info)

#### `MixedPrecisionOptimizer(compute_precision='float32', result_precision='float64')`

Mixed-precision computing optimizer for performance improvement while maintaining accuracy.

**Parameters:**
- `compute_precision: str` - Precision for internal computations (default: 'float32')
- `result_precision: str` - Precision for final results (default: 'float64')

**Methods:**
- `hp_inv_mixed_precision(G, b, ...)` - HP-INV using mixed precision

**Example:**
```python
from src.performance_optimization import MixedPrecisionOptimizer

mixed_opt = MixedPrecisionOptimizer()
x, iters, info = mixed_opt.hp_inv_mixed_precision(G, b, max_iter=10)
```

#### `SparseRRAMModel()`

Sparse matrix implementation for RRAM models to improve performance on large systems with sparse conductance patterns.

**Methods:**
- `create_sparse_rram_matrix(n, sparsity=0.1, ...)` - Create sparse RRAM matrix

**Example:**
```python
from src.performance_optimization import SparseRRAMModel

sparse_model = SparseRRAMModel()
if sparse_model.sparse_available:
    G_sparse = sparse_model.create_sparse_rram_matrix(256, sparsity=0.05)
    print(f"Created sparse matrix with 5% non-zero elements")
```

#### `HardwareTestFixture(hardware_interface, test_mode="simulation")`

Test fixture for hardware integration testing.

**Parameters:**
- `hardware_interface: HardwareInterface` - Interface to hardware device
- `test_mode: str` - Test mode ("simulation" or "real")

**Methods:**
- `run_conductance_test(target_matrix)` - Test conductance programming
- `run_mvm_test(matrix, vector)` - Test matrix-vector multiplication
- `run_inversion_test(matrix)` - Test matrix inversion

**Example:**
```python
from src.hardware_interface import MockRRAMHardware, HardwareTestFixture

# Create a fixture
hardware = MockRRAMHardware(size=4)
fixture = HardwareTestFixture(hardware, test_mode="simulation")

# Run tests
with fixture:
    # Test conductance programming
    test_matrix = np.random.rand(4, 4) * 1e-6 + 0.5 * np.eye(4)
    results = fixture.run_conductance_test(test_matrix)
    print(f"Conductance test passed: {results['passed']}")
```

## Usage Examples

### Complete Example: Solving a Linear System with RRAM Effects

```python
import numpy as np
from src.rram_model import create_rram_matrix
from src.redundancy import apply_redundancy
from src.hp_inv import hp_inv

# Step 1: Create an RRAM matrix with realistic properties
print("Creating RRAM matrix with realistic properties...")
G = create_rram_matrix(
    n=8,
    variability=0.03,      # 3% conductance variability
    stuck_fault_prob=0.01, # 1% stuck-at faults
    line_resistance=1.5e-3 # 1.5 mΩ line resistance
)

# Step 2: Apply redundancy to mitigate stuck faults
print("Applying redundancy to repair faults...")
G_repaired = apply_redundancy(G)
# Ensure the matrix is well-conditioned for inversion
G_repaired = G_repaired + 0.3 * np.eye(8)

# Step 3: Generate right-hand side vector
b = np.random.randn(8)

# Step 4: Solve using HP-INV
print("Solving linear system using HP-INV...")
x_solution, iterations, info = hp_inv(
    G_repaired,
    b,
    bits=4,              # Use 4-bit precision
    lp_noise_std=0.01,   # 1% noise in low-precision inversion
    max_iter=15,         # Allow up to 15 iterations
    tol=1e-6,            # Convergence tolerance
    relaxation_factor=1.0 # Relaxation factor for convergence
)

# Step 5: Validate the solution
residual = np.linalg.norm(G_repaired @ x_solution - b)
print(f"Solution found in {iterations} iterations")
print(f"Final residual: {residual:.2e}")
print(f"Converged: {info['converged']}")
print(f"Condition number: {info['condition_number_estimate']:.2e}")
```

### Benchmarking Different HP-INV Approaches

```python
import numpy as np
from src.rram_model import create_rram_matrix
from src.hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
from src.redundancy import apply_redundancy

# Create test system
G = create_rram_matrix(12, variability=0.02, stuck_fault_prob=0.005)
G = G + 0.4 * np.eye(12)  # Ensure well-conditioned
G = apply_redundancy(G)
b = np.random.randn(12)

# Solve with different approaches
methods = [
    ("Standard HP-INV", lambda: hp_inv(G, b, max_iter=20)),
    ("Block HP-INV", lambda: block_hp_inv(G, b, block_size=6, max_iter=20)),
    ("Adaptive HP-INV", lambda: adaptive_hp_inv(G, b, initial_bits=2, max_iter=20))
]

for name, method in methods:
    x, iters, info = method()
    residual = np.linalg.norm(G @ x - b)
    print(f"{name}: {iters} iterations, residual {residual:.2e}")
```

## Notes

- The Analogico project is designed to simulate RRAM-based analogue computing systems
- Hardware effects like variability, stuck faults, and line resistance are modeled to provide realistic performance estimates
- The HP-INV algorithm achieves high precision through iterative refinement
- BlockAMC enables scaling to larger matrices using smaller physical RRAM arrays
- All algorithms include comprehensive error handling and convergence monitoring

### 9. Material-Specific RRAM Models (`src.advanced_materials_simulation`)

#### `HfO2RRAMModel(device_area=1e-12, temperature=300.0, initial_state="high")`

HfO₂-based RRAM model with specific switching characteristics based on oxygen vacancy migration (VCM mechanism).

**Parameters:**
- `device_area: float` - Physical area of the device in m² (default: 1e-12)
- `temperature: float` - Operating temperature in Kelvin (default: 300.0)
- `initial_state: str` - Initial state "high" or "low" (default: "high")

**Methods:**
- `apply_voltage(voltage, duration)` - Apply voltage with HfO₂-specific VCM switching
- `get_conductance()` - Get current conductance
- `get_device_state()` - Get comprehensive device state

**Example:**
```python
from src.advanced_materials_simulation import HfO2RRAMModel

# Create an HfO2 device model
device = HfO2RRAMModel(device_area=1e-12, temperature=320)

# Apply a SET voltage pulse (formation of oxygen vacancies)
result = device.apply_voltage(-1.8, 1e-8)  # -1.8V, 10ns pulse
print(f"Switched: {result['switched']}, New resistance: {result['resistance_after']:.2e}Ω")
```

#### `TaOxRRAMModel(device_area=1e-12, temperature=300.0, initial_state="high", mechanism="VCM")`

TaOₓ-based RRAM model with VCM or ECM switching mechanism options.

**Parameters:**
- `device_area: float` - Physical area of the device in m² (default: 1e-12)
- `temperature: float` - Operating temperature in Kelvin (default: 300.0)
- `initial_state: str` - Initial state "high" or "low" (default: "high")
- `mechanism: str` - Switching mechanism "VCM" or "ECM" (default: "VCM")

#### `TiO2RRAMModel(device_area=1e-12, temperature=300.0, initial_state="high")`

TiO₂-based RRAM model with specific switching characteristics based on titanium interstitials and oxygen vacancies.

#### `NiOxRRAMModel(device_area=1e-12, temperature=300.0, initial_state="high")`

NiOₓ-based RRAM model with ECM mechanism and metal filament formation.

#### `CoOxRRAMModel(device_area=1e-12, temperature=300.0, initial_state="high")`

CoOₓ-based RRAM model with mixed VCM/ECM behavior.

#### `AdvancedRRAMModel(material=RRAMMaterialType.HFO2, device_area=1e-12, temperature=300.0, initial_resistance_state="high", ecm_vcm_ratio=0.5)`

Main class that provides access to material-specific models with unified interface.

**Parameters:**
- `material: RRAMMaterialType` - Type of RRAM material (default: RRAMMaterialType.HFO2)
- `device_area: float` - Physical area of the device in m² (default: 1e-12)
- `temperature: float` - Operating temperature in Kelvin (default: 300.0)
- `initial_resistance_state: str` - Initial state "high" or "low" (default: "high")
- `ecm_vcm_ratio: float` - Ratio of ECM to VCM for mixed-mechanism materials (default: 0.5)

**Example:**
```python
from src.advanced_rram_models import AdvancedRRAMModel, RRAMMaterialType

# Create devices with different materials
materials = ["HfO2", "TaOx", "TiO2", "NiOx", "CoOx"]

for material in materials:
    device = AdvancedRRAMModel(material=getattr(RRAMMaterialType, material.replace("Ox", "O2")))
    state = device.get_device_state()
    print(f"{material}: {state['resistance']:.2e}Ω, state: {state['state']}")
```