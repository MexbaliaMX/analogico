"""
Dependency resolution and optional module handling for the Analogico project.

This module handles optional dependencies and provides fallback implementations
when certain packages are not available.
"""
import warnings
import numpy as np
from typing import Optional, Any, Dict, Tuple, List

# Core dependencies (required)
CORE_DEPENDENCIES = {
    'numpy': True,
    'scipy': True, 
    'matplotlib': True
}

# Optional dependencies (nice-to-have)
OPTIONAL_DEPENDENCIES = {
    # ML frameworks
    'torch': False,
    'tensorflow': False,
    'jax': False,
    'cupy': False,
    
    # Hardware interfaces
    'serial': False,
    'spidev': False,
    'smbus2': False,
    
    # Advanced features
    'hypothesis': False,
    'pytest': False,
    'coverage': False
}

# Check for and import available dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    OPTIONAL_DEPENDENCIES['torch'] = False
    warnings.warn("PyTorch not available, torch RRAM integration disabled")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    OPTIONAL_DEPENDENCIES['tensorflow'] = False
    warnings.warn("TensorFlow not available, TF RRAM integration disabled")

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    OPTIONAL_DEPENDENCIES['jax'] = False
    warnings.warn("JAX not available, GPU acceleration disabled")

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    OPTIONAL_DEPENDENCIES['cupy'] = False
    warnings.warn("CuPy not available, GPU acceleration disabled")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    OPTIONAL_DEPENDENCIES['serial'] = False
    warnings.warn("PySerial not available, Arduino interface disabled")

try:
    import spidev
    SPI_AVAILABLE = True
except ImportError:
    SPI_AVAILABLE = False
    OPTIONAL_DEPENDENCIES['spidev'] = False
    warnings.warn("spidev not available, SPI communication disabled")

try:
    import smbus2
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    OPTIONAL_DEPENDENCIES['smbus2'] = False
    warnings.warn("smbus2 not available, I2C communication disabled")

try:
    from hypothesis import given, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    OPTIONAL_DEPENDENCIES['hypothesis'] = False
    warnings.warn("Hypothesis not available, property-based testing disabled")

# Now define fallback classes and functions for missing dependencies

# PyTorch fallbacks
if not TORCH_AVAILABLE:
    # Create dummy classes that raise errors when instantiated
    class DummyTorchClass:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Install with: pip install torch")
        
        def __getattr__(self, name):
            raise ImportError("PyTorch is not available. Install with: pip install torch")
    
    # Use these as fallbacks
    class TorchRRAMLinear(DummyTorchClass):
        pass
    
    class TorchRRAMOptimizer(DummyTorchClass):
        pass

# TensorFlow fallbacks
if not TF_AVAILABLE:
    class DummyTFClass:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is not available. Install with: pip install tensorflow")
        
        def __getattr__(self, name):
            raise ImportError("TensorFlow is not available. Install with: pip install tensorflow")
    
    class TensorFRRAMLayer(DummyTFClass):
        pass

# JAX fallbacks
if not JAX_AVAILABLE:
    jnp = np  # Use numpy as fallback
    
    def jit(func):
        """Dummy JIT decorator that just returns the function."""
        return func

# Hardware interface fallbacks
if not SERIAL_AVAILABLE:
    class DummyArduinoInterface:
        """Dummy Arduino interface when pyserial is not available."""
        def __init__(self, *args, **kwargs):
            warnings.warn("Hardware interface not available - using simulation only")
        
        def connect(self):
            return False
        
        def disconnect(self):
            return True
        
        def write_matrix(self, matrix):
            # Just simulate the operation
            return True
        
        def read_matrix(self):
            # Return a default matrix
            return np.eye(8) * 1e-4
        
        def matrix_vector_multiply(self, vector):
            # Just simulate the operation
            return np.random.rand(len(vector)) * 1e-4

# Create utility functions
def check_dependency(dependency_name: str) -> bool:
    """
    Check if a dependency is available.
    
    Args:
        dependency_name: Name of the dependency to check
        
    Returns:
        True if available, False otherwise
    """
    if dependency_name in CORE_DEPENDENCIES:
        return CORE_DEPENDENCIES[dependency_name]
    elif dependency_name in OPTIONAL_DEPENDENCIES:
        return OPTIONAL_DEPENDENCIES[dependency_name]
    else:
        warnings.warn(f"Unknown dependency: {dependency_name}")
        return False

def require_dependency(dependency_name: str, feature_name: str = "feature"):
    """
    Decorator to require a dependency for a function.
    
    Args:
        dependency_name: Name of required dependency
        feature_name: Name of the feature requiring the dependency
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_dependency(dependency_name):
                raise ImportError(
                    f"{dependency_name} is required for {feature_name}. "
                    f"Install with: pip install {dependency_name}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_array_library():
    """
    Get the best available array library (JAX -> NumPy -> CuPy).
    
    Returns:
        Array library module
    """
    if JAX_AVAILABLE:
        return jax.numpy
    elif CUPY_AVAILABLE:
        return cupy
    else:
        return np

def handle_missing_dependency(dependency_name: str, 
                             install_instruction: str = None) -> Any:
    """
    Create a handler for missing dependencies that raises informative errors.
    
    Args:
        dependency_name: Name of the missing dependency
        install_instruction: Custom installation instruction (optional)
        
    Returns:
        Handler that raises ImportError when accessed
    """
    if install_instruction is None:
        install_instruction = f"pip install {dependency_name}"
    
    def handler(*args, **kwargs):
        raise ImportError(
            f"{dependency_name} is required but not available. "
            f"Install with: {install_instruction}"
        )
    
    return handler

# Export available dependencies status
DEPENDENCY_STATUS = {
    'torch': TORCH_AVAILABLE,
    'tensorflow': TF_AVAILABLE,
    'jax': JAX_AVAILABLE,
    'cupy': CUPY_AVAILABLE,
    'serial': SERIAL_AVAILABLE,
    'spidev': SPI_AVAILABLE,
    'smbus2': I2C_AVAILABLE,
    'hypothesis': HYPOTHESIS_AVAILABLE
}

# Define module availability for use throughout the codebase
MODULES_AVAILABLE = {
    'ml_integration': TORCH_AVAILABLE or TF_AVAILABLE,
    'gpu_acceleration': JAX_AVAILABLE or CUPY_AVAILABLE,
    'hardware_interface': SERIAL_AVAILABLE or SPI_AVAILABLE or I2C_AVAILABLE,
    'advanced_testing': HYPOTHESIS_AVAILABLE
}

def print_dependency_report():
    """
    Print a report of available and missing dependencies.
    """
    print("Dependency Report for Analogico Project")
    print("=" * 40)
    print("Required Dependencies:")
    for dep, available in CORE_DEPENDENCIES.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {dep}: {status}")
    
    print("\nOptional Dependencies:")
    for dep, available in OPTIONAL_DEPENDENCIES.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {dep}: {status}")
    
    print("\nModule Availability:")
    for module, available in MODULES_AVAILABLE.items():
        status = "✓ Available" if available else "✗ Limited"
        print(f"  {module}: {status}")


def get_fallback_implementation(class_name: str):
    """
    Get a fallback implementation for a class when dependencies are missing.
    
    Args:
        class_name: Name of the class to get fallback for
        
    Returns:
        Fallback implementation
    """
    # Common fallback implementations
    fallbacks = {
        'TorchRRAMLinear': TorchRRAMLinear if TORCH_AVAILABLE else DummyTorchClass,
        'TorchRRAMOptimizer': TorchRRAMOptimizer if TORCH_AVAILABLE else DummyTorchClass,
        'TensorFRRAMLayer': TensorFRRAMLayer if TF_AVAILABLE else DummyTFClass,
        'ArduinoRRAMInterface': DummyArduinoInterface if not SERIAL_AVAILABLE else None
    }
    
    return fallbacks.get(class_name, None)


# At module level, ensure all key variables are defined
__all__ = [
    'TORCH_AVAILABLE',
    'TF_AVAILABLE', 
    'JAX_AVAILABLE',
    'CUPY_AVAILABLE',
    'SERIAL_AVAILABLE',
    'SPI_AVAILABLE',
    'I2C_AVAILABLE',
    'HYPOTHESIS_AVAILABLE',
    'DEPENDENCY_STATUS',
    'MODULES_AVAILABLE',
    'check_dependency',
    'require_dependency', 
    'get_array_library',
    'handle_missing_dependency',
    'print_dependency_report',
    'get_fallback_implementation',
    'TorchRRAMLinear',
    'TorchRRAMOptimizer',
    'TensorFRRAMLayer'
]

# Initialize the dependency status at import time
if __name__ == "__main__":
    print_dependency_report()