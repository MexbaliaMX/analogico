"""
GPU-Accelerated Simulation for RRAM Systems.

This module provides GPU acceleration for RRAM simulations and algorithms,
enabling faster computation of large-scale RRAM models and algorithms.
"""
import numpy as np
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass
import time

# Try to import JAX for GPU acceleration
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad, pmap, jacfwd, jacrev
    from jax.experimental import sparse as jsparse
    from jax.tree_util import tree_map
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available, GPU acceleration disabled")
    jnp = np  # Fallback to numpy

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available, GPU acceleration disabled")

# Try to import PyTorch with CUDA support
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False
    warnings.warn("PyTorch not available, GPU acceleration disabled")

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV, AdaptivePrecisionHPINV
    from .neuromorphic_modules import RRAMSpikingLayer, RRAMReservoirComputer
    from .optimization_algorithms import RRAMAwareOptimizer
    from .benchmarking_suite import RRAMBenchmarkSuite
    from .advanced_materials_simulation import AdvancedMaterialsSimulator
    from .edge_computing_integration import EdgeRRAMOptimizer
    from .performance_profiling import PerformanceProfiler
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class GPUSimulationAccelerator:
    """
    GPU-accelerated simulation for RRAM systems.
    """
    
    def __init__(self, 
                 backend: str = 'jax',
                 device_memory_limit: Optional[float] = None):
        """
        Initialize the GPU simulation accelerator.
        
        Args:
            backend: Acceleration backend ('jax', 'cupy', 'torch')
            device_memory_limit: Memory limit for GPU in GB (None for no limit)
        """
        self.backend = backend
        self.device_memory_limit = device_memory_limit
        self.xp = self._get_array_module()
        self.initialized = self._initialize_backend()
    
    def _get_array_module(self):
        """Get the appropriate array module based on availability and preference."""
        if self.backend == 'jax' and JAX_AVAILABLE:
            return jnp
        elif self.backend == 'cupy' and CUPY_AVAILABLE:
            return cp
        elif self.backend == 'torch' and TORCH_AVAILABLE:
            return torch
        else:
            # Fallback to numpy
            return np
    
    def _initialize_backend(self) -> bool:
        """Initialize the chosen backend."""
        try:
            if self.backend == 'jax' and JAX_AVAILABLE:
                # JAX initialization
                print(f"Using JAX backend with {jax.device_count()} devices")
                return True
            elif self.backend == 'cupy' and CUPY_AVAILABLE:
                # CuPy initialization
                print(f"Using CuPy backend with {cp.cuda.runtime.getDeviceCount()} devices")
                return True
            elif self.backend == 'torch' and TORCH_AVAILABLE:
                # PyTorch initialization
                device = 'cuda' if TORCH_CUDA_AVAILABLE else 'cpu'
                print(f"Using PyTorch backend on {device}")
                return True
            else:
                print("Falling back to CPU-only simulation")
                return False
        except Exception as e:
            warnings.warn(f"Backend initialization failed: {e}")
            return False
    
    def gpu_hp_inv(self, 
                   G: np.ndarray, 
                   b: np.ndarray, 
                   **kwargs) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        GPU-accelerated HP-INV for solving Gx = b.
        
        Args:
            G: Conductance matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (solution x, iterations taken, convergence info dict)
        """
        if not self.initialized:
            # Fallback to CPU if GPU not available
            return hp_inv(G, b, **kwargs)
        
        # Convert to appropriate GPU array type
        if self.backend == 'jax':
            return self._gpu_hp_inv_jax(G, b, **kwargs)
        elif self.backend == 'cupy':
            return self._gpu_hp_inv_cupy(G, b, **kwargs)
        elif self.backend == 'torch':
            return self._gpu_hp_inv_torch(G, b, **kwargs)
        else:
            return hp_inv(G, b, **kwargs)
    
    def _gpu_hp_inv_jax(self, 
                        G: np.ndarray, 
                        b: np.ndarray, 
                        **kwargs) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """JAX implementation of HP-INV."""
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available")
        
        # Convert to JAX arrays
        G_jax = jnp.array(G)
        b_jax = jnp.array(b)
        
        bits = kwargs.get('bits', 3)
        max_iter = kwargs.get('max_iter', 20)
        tol = kwargs.get('tol', 1e-6)
        lp_noise_std = kwargs.get('lp_noise_std', 0.01)
        relaxation_factor = kwargs.get('relaxation_factor', 1.0)
        
        def quantize(matrix, bits):
            """Quantize matrix to given bit precision."""
            if matrix.size == 0 or jnp.all(matrix == 0):
                return matrix
            max_val = jnp.max(jnp.abs(matrix))
            levels = 2**bits - 1
            if max_val == 0:
                return matrix
            scale = levels / max_val
            quantized = jnp.round(matrix * scale) / scale
            return quantized

        # LP-INV: Quantize and invert with noise
        G_lp = quantize(G_jax, bits)

        try:
            G_lp_inv = jnp.linalg.inv(G_lp)
            noise = jax.random.normal(jax.random.PRNGKey(42), G_lp_inv.shape) * lp_noise_std
            A0_inv = G_lp_inv + noise
        except Exception:
            # Singular matrix, return zero correction
            A0_inv = jnp.zeros_like(G_lp)
            warnings.warn("Matrix is singular, using zero inverse approximation")

        # Initialize solution
        x = jnp.zeros_like(b_jax, dtype=float)
        residuals = []

        # JIT-compiled iterative refinement loop
        @jax.jit
        def refine_step(x, A0_inv, G, b, relaxation_factor):
            # Compute residual r = b - G x
            Ax = G @ x
            r = b - Ax
            residual_norm = jnp.linalg.norm(r)
            
            # Update x with relaxation factor
            delta = A0_inv @ r
            x_new = x + relaxation_factor * delta
            
            return x_new, residual_norm

        # Run iterative refinement
        for k in range(max_iter):
            x, residual_norm = refine_step(x, A0_inv, G_jax, b_jax, relaxation_factor)
            residuals.append(float(residual_norm))
            
            # Check convergence
            if residual_norm < tol:
                break

        # Convert back to numpy arrays
        x_result = np.array(x)
        residuals = [float(r) for r in residuals]

        info = {
            'residuals': residuals,
            'final_residual': residuals[-1] if residuals else 0,
            'converged': residuals[-1] < tol if residuals else False,
            'condition_number_estimate': None
        }

        # Calculate condition number if matrix is invertible
        try:
            info['condition_number_estimate'] = np.linalg.cond(G)
        except:
            info['condition_number_estimate'] = float('inf')

        return x_result, len(residuals), info
    
    def _gpu_hp_inv_cupy(self, 
                         G: np.ndarray, 
                         b: np.ndarray, 
                         **kwargs) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """CuPy implementation of HP-INV."""
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not available")
        
        # Convert to CuPy arrays
        G_cp = cp.array(G)
        b_cp = cp.array(b)
        
        bits = kwargs.get('bits', 3)
        max_iter = kwargs.get('max_iter', 20)
        tol = kwargs.get('tol', 1e-6)
        lp_noise_std = kwargs.get('lp_noise_std', 0.01)
        relaxation_factor = kwargs.get('relaxation_factor', 1.0)
        
        def quantize(matrix, bits):
            """Quantize matrix to given bit precision."""
            if matrix.size == 0 or cp.all(matrix == 0):
                return matrix
            max_val = cp.max(cp.abs(matrix))
            levels = 2**bits - 1
            if max_val == 0:
                return matrix
            scale = levels / max_val
            quantized = cp.round(matrix * scale) / scale
            return quantized

        # LP-INV: Quantize and invert with noise
        G_lp = quantize(G_cp, bits)

        try:
            G_lp_inv = cp.linalg.inv(G_lp)
            noise = cp.random.normal(0, lp_noise_std, G_lp_inv.shape)
            A0_inv = G_lp_inv + noise
        except cp.linalg.LinAlgError:
            # Singular matrix, return zero correction
            A0_inv = cp.zeros_like(G_lp)
            warnings.warn("Matrix is singular, using zero inverse approximation")

        # Initialize solution
        x = cp.zeros_like(b_cp, dtype=cp.float64)
        residuals = []

        for k in range(max_iter):
            # Compute residual r = b - G x
            Ax = G_cp @ x
            r = b_cp - Ax
            residual_norm = cp.linalg.norm(r)
            residuals.append(float(residual_norm))

            # Update x with relaxation factor
            delta = A0_inv @ r
            x += relaxation_factor * delta

            # Check convergence
            if residual_norm < tol:
                break

        # Convert back to numpy arrays
        x_result = cp.asnumpy(x)
        residuals = [float(r) for r in residuals]

        info = {
            'residuals': residuals,
            'final_residual': residuals[-1] if residuals else 0,
            'converged': residuals[-1] < tol if residuals else False,
            'condition_number_estimate': None
        }

        # Calculate condition number if matrix is invertible
        try:
            info['condition_number_estimate'] = np.linalg.cond(G)
        except:
            info['condition_number_estimate'] = float('inf')

        return x_result, len(residuals), info
    
    def _gpu_hp_inv_torch(self, 
                          G: np.ndarray, 
                          b: np.ndarray, 
                          **kwargs) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """PyTorch implementation of HP-INV."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
        
        device = torch.device('cuda' if TORCH_CUDA_AVAILABLE else 'cpu')
        
        # Convert to PyTorch tensors
        G_torch = torch.tensor(G, dtype=torch.float64, device=device)
        b_torch = torch.tensor(b, dtype=torch.float64, device=device)
        
        bits = kwargs.get('bits', 3)
        max_iter = kwargs.get('max_iter', 20)
        tol = kwargs.get('tol', 1e-6)
        lp_noise_std = kwargs.get('lp_noise_std', 0.01)
        relaxation_factor = kwargs.get('relaxation_factor', 1.0)
        
        def quantize(matrix, bits):
            """Quantize matrix to given bit precision."""
            if matrix.numel() == 0 or torch.all(matrix == 0):
                return matrix
            max_val = torch.max(torch.abs(matrix))
            levels = 2**bits - 1
            if max_val == 0:
                return matrix
            scale = levels / max_val
            quantized = torch.round(matrix * scale) / scale
            return quantized

        # LP-INV: Quantize and invert with noise
        G_lp = quantize(G_torch, bits)

        try:
            G_lp_inv = torch.inverse(G_lp)
            noise = torch.randn(G_lp_inv.shape, device=device, dtype=torch.float64) * lp_noise_std
            A0_inv = G_lp_inv + noise
        except torch._C._LinAlgError:
            # Singular matrix, return zero correction
            A0_inv = torch.zeros_like(G_lp)
            warnings.warn("Matrix is singular, using zero inverse approximation")

        # Initialize solution
        x = torch.zeros_like(b_torch, dtype=torch.float64)
        residuals = []

        for k in range(max_iter):
            # Compute residual r = b - G x
            Ax = G_torch @ x
            r = b_torch - Ax
            residual_norm = torch.norm(r)
            residuals.append(residual_norm.item())

            # Update x with relaxation factor
            delta = A0_inv @ r
            x += relaxation_factor * delta

            # Check convergence
            if residual_norm < tol:
                break

        # Convert back to numpy arrays
        x_result = x.cpu().numpy()
        residuals = [float(r) for r in residuals]

        info = {
            'residuals': residuals,
            'final_residual': residuals[-1] if residuals else 0,
            'converged': residuals[-1] < tol if residuals else False,
            'condition_number_estimate': None
        }

        # Calculate condition number if matrix is invertible
        try:
            info['condition_number_estimate'] = np.linalg.cond(G)
        except:
            info['condition_number_estimate'] = float('inf')

        return x_result, len(residuals), info
    
    def gpu_material_simulation(self, 
                                material_type: str, 
                                num_devices: int = 1000,
                                **kwargs) -> np.ndarray:
        """
        GPU-accelerated simulation of RRAM material properties for many devices.
        
        Args:
            material_type: Type of RRAM material ('HfO2', 'TaOx', etc.)
            num_devices: Number of devices to simulate
            **kwargs: Additional parameters
            
        Returns:
            Array of simulated device properties
        """
        if not self.initialized:
            # Fallback to CPU simulation
            return self._cpu_material_simulation(material_type, num_devices, **kwargs)
        
        start_time = time.time()
        
        if self.backend == 'jax':
            results = self._gpu_material_simulation_jax(material_type, num_devices, **kwargs)
        elif self.backend == 'cupy':
            results = self._gpu_material_simulation_cupy(material_type, num_devices, **kwargs)
        elif self.backend == 'torch':
            results = self._gpu_material_simulation_torch(material_type, num_devices, **kwargs)
        else:
            results = self._cpu_material_simulation(material_type, num_devices, **kwargs)
        
        print(f"Material simulation completed: {num_devices} devices in {time.time() - start_time:.4f}s")
        
        return results
    
    def _gpu_material_simulation_jax(self, material_type: str, num_devices: int, **kwargs) -> np.ndarray:
        """JAX-based material simulation."""
        if not JAX_AVAILABLE:
            return self._cpu_material_simulation(material_type, num_devices, **kwargs)
        
        # Create device parameters
        initial_conductances = jnp.random.uniform(1e-6, 1e-3, size=(num_devices,))
        variabilities = jnp.random.normal(0.05, 0.01, size=(num_devices,))
        temperatures = jnp.random.uniform(300, 400, size=(num_devices,))
        
        # Apply material-specific characteristics
        if material_type == 'HfO2':
            # HfO2-specific properties
            activation_energy = jnp.full(num_devices, 0.65)
        elif material_type == 'TaOx':
            activation_energy = jnp.full(num_devices, 0.75)
        elif material_type == 'TiO2':
            activation_energy = jnp.full(num_devices, 0.80)
        else:
            activation_energy = jnp.full(num_devices, 0.70)
        
        # Simulate temperature effects
        k_boltzmann = 8.617e-5  # eV/K
        temp_factors = jnp.exp(activation_energy * (1/300 - 1/temperatures) / k_boltzmann)
        conductances_with_temp = initial_conductances * (1 + 0.1 * temp_factors)
        
        # Add variability
        final_conductances = conductances_with_temp * (1 + variabilities)
        
        return np.array(final_conductances)
    
    def _gpu_material_simulation_cupy(self, material_type: str, num_devices: int, **kwargs) -> np.ndarray:
        """CuPy-based material simulation."""
        if not CUPY_AVAILABLE:
            return self._cpu_material_simulation(material_type, num_devices, **kwargs)
        
        # Create device parameters
        initial_conductances = cp.random.uniform(1e-6, 1e-3, size=(num_devices,))
        variabilities = cp.random.normal(0.05, 0.01, size=(num_devices,))
        temperatures = cp.random.uniform(300, 400, size=(num_devices,))
        
        # Apply material-specific characteristics
        if material_type == 'HfO2':
            activation_energy = cp.full(num_devices, 0.65)
        elif material_type == 'TaOx':
            activation_energy = cp.full(num_devices, 0.75)
        elif material_type == 'TiO2':
            activation_energy = cp.full(num_devices, 0.80)
        else:
            activation_energy = cp.full(num_devices, 0.70)
        
        # Simulate temperature effects
        k_boltzmann = 8.617e-5  # eV/K
        temp_factors = cp.exp(activation_energy * (1/300 - 1/temperatures) / k_boltzmann)
        conductances_with_temp = initial_conductances * (1 + 0.1 * temp_factors)
        
        # Add variability
        final_conductances = conductances_with_temp * (1 + variabilities)
        
        return cp.asnumpy(final_conductances)
    
    def _gpu_material_simulation_torch(self, material_type: str, num_devices: int, **kwargs) -> np.ndarray:
        """PyTorch-based material simulation."""
        if not TORCH_AVAILABLE:
            return self._cpu_material_simulation(material_type, num_devices, **kwargs)
        
        device = torch.device('cuda' if TORCH_CUDA_AVAILABLE else 'cpu')
        
        # Create device parameters
        initial_conductances = torch.rand(num_devices, device=device, dtype=torch.float64) * 9e-4 + 1e-6
        variabilities = torch.normal(0.05, 0.01, size=(num_devices,), device=device, dtype=torch.float64)
        temperatures = torch.rand(num_devices, device=device, dtype=torch.float64) * 100 + 300
        
        # Apply material-specific characteristics
        if material_type == 'HfO2':
            activation_energy = torch.full((num_devices,), 0.65, device=device, dtype=torch.float64)
        elif material_type == 'TaOx':
            activation_energy = torch.full((num_devices,), 0.75, device=device, dtype=torch.float64)
        elif material_type == 'TiO2':
            activation_energy = torch.full((num_devices,), 0.80, device=device, dtype=torch.float64)
        else:
            activation_energy = torch.full((num_devices,), 0.70, device=device, dtype=torch.float64)
        
        # Simulate temperature effects
        k_boltzmann = 8.617e-5  # eV/K
        temp_factors = torch.exp(activation_energy * (1/300 - 1/temperatures) / k_boltzmann)
        conductances_with_temp = initial_conductances * (1 + 0.1 * temp_factors)
        
        # Add variability
        final_conductances = conductances_with_temp * (1 + variabilities)
        
        return final_conductances.cpu().numpy()
    
    def _cpu_material_simulation(self, material_type: str, num_devices: int, **kwargs) -> np.ndarray:
        """CPU fallback for material simulation."""
        # Create device parameters
        initial_conductances = np.random.uniform(1e-6, 1e-3, size=num_devices)
        variabilities = np.random.normal(0.05, 0.01, size=num_devices)
        temperatures = np.random.uniform(300, 400, size=num_devices)
        
        # Apply material-specific characteristics
        if material_type == 'HfO2':
            activation_energy = np.full(num_devices, 0.65)
        elif material_type == 'TaOx':
            activation_energy = np.full(num_devices, 0.75)
        elif material_type == 'TiO2':
            activation_energy = np.full(num_devices, 0.80)
        else:
            activation_energy = np.full(num_devices, 0.70)
        
        # Simulate temperature effects
        k_boltzmann = 8.617e-5  # eV/K
        temp_factors = np.exp(activation_energy * (1/300 - 1/temperatures) / k_boltzmann)
        conductances_with_temp = initial_conductances * (1 + 0.1 * temp_factors)
        
        # Add variability
        final_conductances = conductances_with_temp * (1 + variabilities)
        
        return final_conductances
    
    def benchmark_gpu_acceleration(self, 
                                  matrix_sizes: List[int] = [10, 50, 100, 200],
                                  num_trials: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Benchmark GPU acceleration performance against CPU.
        
        Args:
            matrix_sizes: List of matrix sizes to test
            num_trials: Number of trials for averaging
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'cpu_times': {},
            'gpu_times': {},
            'speedups': {}
        }
        
        for size in matrix_sizes:
            cpu_times = []
            gpu_times = []
            
            for trial in range(num_trials):
                # Create test problem
                G = np.random.rand(size, size) * 1e-4
                G = G + 0.5 * np.eye(size)  # Diagonally dominant
                b = np.random.rand(size)
                
                # CPU benchmark
                start_time = time.time()
                cpu_solution, cpu_iter, cpu_info = hp_inv(G, b, max_iter=10, bits=4)
                cpu_time = time.time() - start_time
                cpu_times.append(cpu_time)
                
                # GPU benchmark
                if self.initialized:
                    start_time = time.time()
                    gpu_solution, gpu_iter, gpu_info = self.gpu_hp_inv(G, b, max_iter=10, bits=4)
                    gpu_time = time.time() - start_time
                    gpu_times.append(gpu_time)
                else:
                    gpu_times.append(float('inf'))  # GPU not available
            
            avg_cpu_time = np.mean(cpu_times)
            avg_gpu_time = np.mean(gpu_times) if gpu_times else float('inf')
            
            results['cpu_times'][size] = avg_cpu_time
            results['gpu_times'][size] = avg_gpu_time
            results['speedups'][size] = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
        
        return results


class SimulationAccelerationManager:
    """
    Manager for simulation acceleration strategies.
    """
    
    def __init__(self):
        self.accelerators = {}
        self.default_backend = 'jax' if JAX_AVAILABLE else 'torch' if TORCH_AVAILABLE else 'cupy' if CUPY_AVAILABLE else 'cpu'
        self.global_accelerator = None
    
    def initialize_accelerator(self, 
                              backend: str = None,
                              device_memory_limit: Optional[float] = None) -> GPUSimulationAccelerator:
        """
        Initialize a simulation accelerator.
        
        Args:
            backend: Backend to use ('jax', 'cupy', 'torch', 'cpu')
            device_memory_limit: Memory limit for GPU in GB
            
        Returns:
            Initialized accelerator instance
        """
        if backend is None:
            backend = self.default_backend
        
        accel_key = f"{backend}_{device_memory_limit}"
        if accel_key not in self.accelerators:
            self.accelerators[accel_key] = GPUSimulationAccelerator(
                backend=backend,
                device_memory_limit=device_memory_limit
            )
        
        if self.global_accelerator is None:
            self.global_accelerator = self.accelerators[accel_key]
        
        return self.accelerators[accel_key]
    
    def select_optimal_accelerator(self, 
                                  problem_type: str = 'linear',
                                  problem_size: int = 100) -> GPUSimulationAccelerator:
        """
        Select the optimal accelerator based on problem type and size.
        
        Args:
            problem_type: Type of problem ('linear', 'material', 'neural')
            problem_size: Size of the problem
            
        Returns:
            Optimal accelerator instance
        """
        # Simple heuristic for now - in reality, this could be much more sophisticated
        if problem_size > 500:
            # Use JAX for large problems (better optimization and compilation)
            preferred_backends = ['jax', 'cupy', 'torch']
        elif problem_type == 'material':
            # Material simulation might benefit from different backends
            preferred_backends = ['cupy', 'jax', 'torch']
        else:
            # For smaller problems, any backend should work
            preferred_backends = [self.default_backend, 'jax', 'torch', 'cupy']
        
        # Try each preferred backend in order
        for backend in preferred_backends:
            if (backend == 'jax' and JAX_AVAILABLE) or \
               (backend == 'cupy' and CUPY_AVAILABLE) or \
               (backend == 'torch' and TORCH_AVAILABLE) or \
               backend == 'cpu':
                return self.initialize_accelerator(backend=backend)
        
        # Fallback to CPU-only
        return self.initialize_accelerator(backend='cpu')
    
    def accelerate_linear_system(self, 
                                G: np.ndarray, 
                                b: np.ndarray,
                                accelerator: Optional[GPUSimulationAccelerator] = None,
                                **kwargs) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Accelerate solution of linear system using optimal accelerator.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            accelerator: Pre-selected accelerator (optional)
            **kwargs: Additional parameters
            
        Returns:
            Solution from accelerated solver
        """
        if accelerator is None:
            accelerator = self.select_optimal_accelerator(
                problem_type='linear',
                problem_size=G.shape[0]
            )
        
        return accelerator.gpu_hp_inv(G, b, **kwargs)
    
    def accelerate_material_simulation(self,
                                     material_type: str,
                                     num_devices: int,
                                     accelerator: Optional[GPUSimulationAccelerator] = None,
                                     **kwargs) -> np.ndarray:
        """
        Accelerate material simulation using optimal accelerator.
        
        Args:
            material_type: RRAM material type
            num_devices: Number of devices to simulate
            accelerator: Pre-selected accelerator (optional)
            **kwargs: Additional parameters
            
        Returns:
            Array of simulated device properties
        """
        if accelerator is None:
            accelerator = self.select_optimal_accelerator(
                problem_type='material',
                problem_size=num_devices
            )
        
        return accelerator.gpu_material_simulation(
            material_type=material_type,
            num_devices=num_devices,
            **kwargs
        )


def create_gpu_accelerator(backend: str = 'jax', 
                          device_memory_limit: Optional[float] = None) -> GPUSimulationAccelerator:
    """
    Factory function to create GPU simulation accelerator.
    
    Args:
        backend: Backend to use ('jax', 'cupy', 'torch')
        device_memory_limit: Memory limit in GB
        
    Returns:
        GPUSimulationAccelerator instance
    """
    return GPUSimulationAccelerator(backend=backend, device_memory_limit=device_memory_limit)


def demonstrate_gpu_acceleration():
    """
    Demonstrate GPU acceleration capabilities.
    """
    print("GPU Acceleration Demonstration")
    print("="*40)
    
    # Create accelerator manager
    manager = SimulationAccelerationManager()
    
    print(f"Available backends: JAX={JAX_AVAILABLE}, CuPy={CUPY_AVAILABLE}, Torch={TORCH_AVAILABLE}")
    
    # Initialize accelerator
    accelerator = manager.initialize_accelerator(backend='jax' if JAX_AVAILABLE else 'torch')
    print(f"Initialized accelerator with backend: {accelerator.backend}")
    print(f"GPU available: {accelerator.initialized}")
    
    # Test with small problem
    print("\nTesting with small linear system (10x10)...")
    G_small = np.random.rand(10, 10) * 1e-4
    G_small = G_small + 0.5 * np.eye(10)
    b_small = np.random.rand(10)
    
    # Solve with CPU
    start_time = time.time()
    cpu_result, cpu_iter, cpu_info = hp_inv(G_small, b_small, max_iter=10, bits=4)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f}s, Iterations: {cpu_iter}")
    
    # Solve with GPU
    if accelerator.initialized:
        start_time = time.time()
        gpu_result, gpu_iter, gpu_info = accelerator.gpu_hp_inv(G_small, b_small, max_iter=10, bits=4)
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f}s, Iterations: {gpu_iter}")
        
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"Speedup: {speedup:.2f}x")
        
        # Compare results
        result_diff = np.linalg.norm(cpu_result - gpu_result)
        print(f"Result difference: {result_diff:.2e}")
    else:
        print("GPU accelerator not initialized, skipping GPU test")
    
    # Test material simulation
    print(f"\nTesting material simulation for 1000 devices...")
    start_time = time.time()
    material_results = accelerator.gpu_material_simulation('HfO2', 1000)
    material_time = time.time() - start_time
    print(f"Material simulation completed in {material_time:.4f}s for 1000 devices")
    print(f"Mean conductance: {np.mean(material_results):.2e} S")
    
    # Benchmark different matrix sizes
    print(f"\nBenchmarking different matrix sizes...")
    if accelerator.initialized:
        benchmark_results = accelerator.benchmark_gpu_acceleration(
            matrix_sizes=[10, 50, 100],
            num_trials=3
        )
        
        for size, speedup in benchmark_results['speedups'].items():
            if speedup > 0:
                print(f"Size {size}x{size}: {speedup:.2f}x speedup")
            else:
                print(f"Size {size}x{size}: GPU not available")
    
    print("\nGPU acceleration demonstration completed!")


if __name__ == "__main__":
    demonstrate_gpu_acceleration()