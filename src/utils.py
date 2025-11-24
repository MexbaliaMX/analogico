"""
Utility functions for the Analogico project.

This module contains common utilities for input validation, quantization,
and other shared functionality across the project.
"""
import numpy as np
from typing import Union, List, Tuple
import warnings


# Numerical stability constants
NUMERICAL_EPSILON = 1e-15  # Threshold for considering values as zero
MIN_SCALE_THRESHOLD = 1e-12  # Minimum scale factor to avoid numerical instability in quantization


def validate_matrix_vector_inputs(
    G: np.ndarray,
    b: np.ndarray,
    func_name: str = "function"
) -> None:
    """
    Validate that G is a 2D square matrix and b is a compatible 1D vector.

    Args:
        G: A 2D matrix
        b: A 1D vector 
        func_name: Name of the calling function for error messages

    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If dimensions don't match or inputs are invalid
    """
    # Validate input types
    if not isinstance(G, np.ndarray):
        raise TypeError(f"G must be a numpy array in {func_name}, got {type(G)}")
    if not isinstance(b, np.ndarray):
        raise TypeError(f"b must be a numpy array in {func_name}, got {type(b)}")

    # Validate dimensions
    if G.ndim != 2:
        raise ValueError(
            f"G must be a 2D matrix in {func_name}, got shape {G.shape} with {G.ndim} dimensions"
        )
    if b.ndim != 1:
        raise ValueError(
            f"b must be a 1D vector in {func_name}, got shape {b.shape} with {b.ndim} dimensions"
        )
    if G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square in {func_name}, got shape {G.shape}")
    if G.shape[0] != b.shape[0]:
        raise ValueError(
            f"Dimension mismatch in {func_name}: G is {G.shape}, b is {b.shape}"
        )

    # Validate numeric values
    if not np.all(np.isfinite(G)):
        raise ValueError(f"G contains non-finite values (NaN or Inf) in {func_name}")
    if not np.all(np.isfinite(b)):
        raise ValueError(f"b contains non-finite values (NaN or Inf) in {func_name}")


def validate_matrix_inputs(
    G: np.ndarray,
    allow_empty: bool = False,
    func_name: str = "function"
) -> None:
    """
    Validate that G is a 2D matrix with valid numeric values.

    Args:
        G: A 2D matrix
        allow_empty: Whether to allow empty matrices
        func_name: Name of the calling function for error messages

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If dimensions are invalid or inputs contain invalid values
    """
    # Validate input type
    if not isinstance(G, np.ndarray):
        raise TypeError(f"G must be a numpy array in {func_name}, got {type(G)}")

    # Validate dimensions
    if G.ndim != 2:
        raise ValueError(
            f"G must be a 2D matrix in {func_name}, got shape {G.shape} with {G.ndim} dimensions"
        )
    
    if not allow_empty and G.shape[0] == 0:
        raise ValueError(f"G cannot be empty in {func_name}")

    # Validate numeric values
    if not np.all(np.isfinite(G)):
        raise ValueError(f"G contains non-finite values (NaN or Inf) in {func_name}")


def validate_vector_inputs(
    x: np.ndarray,
    func_name: str = "function"
) -> None:
    """
    Validate that x is a 1D vector with valid numeric values.

    Args:
        x: A 1D vector
        func_name: Name of the calling function for error messages

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If dimensions are invalid or inputs contain invalid values
    """
    # Validate input type
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a numpy array in {func_name}, got {type(x)}")

    # Validate dimensions
    if x.ndim != 1:
        raise ValueError(
            f"x must be a 1D vector in {func_name}, got shape {x.shape} with {x.ndim} dimensions"
        )

    # Validate numeric values
    if not np.all(np.isfinite(x)):
        raise ValueError(f"x contains non-finite values (NaN or Inf) in {func_name}")


def validate_parameter(
    value: Union[int, float],
    name: str,
    min_value: Union[int, float] = None,
    max_value: Union[int, float] = None,
    integer: bool = False,
    positive: bool = False,
    func_name: str = "function"
) -> None:
    """
    Validate a parameter value against specified constraints.

    Args:
        value: The value to validate
        name: Name of the parameter for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        integer: Whether the value must be an integer
        positive: Whether the value must be positive
        func_name: Name of the calling function for error messages

    Raises:
        TypeError: If value is not of the expected type
        ValueError: If value doesn't meet the constraints
    """
    if integer and not isinstance(value, int):
        raise TypeError(f"{name} must be an integer in {func_name}, got {type(value)}")
    
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric in {func_name}, got {type(value)}")
    
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive in {func_name}, got {value}")
    
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value} in {func_name}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value} in {func_name}, got {value}")


def quantize(matrix: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantize matrix to given bit precision.

    Args:
        matrix: Matrix to quantize
        bits: Number of bits for quantization

    Returns:
        Quantized matrix

    Raises:
        ValueError: If bits is not positive
    """
    if bits <= 0:
        raise ValueError(f"Number of bits must be positive, got {bits}")

    if matrix.size == 0 or np.all(matrix == 0):
        return matrix

    max_val = np.max(np.abs(matrix))

    # Check for zero or near-zero maximum value to prevent division by zero
    if max_val < NUMERICAL_EPSILON:
        warnings.warn(
            f"Matrix maximum value {max_val} is below numerical epsilon {NUMERICAL_EPSILON}, "
            "returning original matrix"
        )
        return matrix

    levels = 2**bits - 1
    scale = levels / max_val

    # Avoid potential numerical issues if scale is too small
    if scale < MIN_SCALE_THRESHOLD:
        warnings.warn(
            f"Scale factor {scale} below threshold {MIN_SCALE_THRESHOLD}, "
            "returning zero matrix to avoid numerical instability"
        )
        return np.zeros_like(matrix)

    # Additional safety check for non-finite scale
    if not np.isfinite(scale):
        warnings.warn(f"Non-finite scale factor: {scale}, returning zero matrix")
        return np.zeros_like(matrix)

    quantized = np.round(matrix * scale) / scale
    return quantized


def matrix_vector_multiply(G: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Perform matrix-vector multiplication with validation.

    Args:
        G: Conductance matrix (m x n)
        x: Input vector (n,)

    Returns:
        Output vector y = G @ x (m,)

    Raises:
        ValueError: If dimensions don't match or inputs are invalid
        TypeError: If inputs are not numpy arrays
    """
    # Validate input types
    if not isinstance(G, np.ndarray):
        raise TypeError(f"G must be a numpy array, got {type(G)}")
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a numpy array, got {type(x)}")

    # Validate dimensions
    if G.ndim != 2:
        raise ValueError(f"G must be a 2D matrix, got shape {G.shape} with {G.ndim} dimensions")
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D vector, got shape {x.shape} with {x.ndim} dimensions")

    # Validate size compatibility
    if G.shape[1] != x.shape[0]:
        raise ValueError(
            f"Matrix-vector dimension mismatch: G has shape {G.shape}, "
            f"x has shape {x.shape}. G.shape[1] ({G.shape[1]}) must equal "
            f"x.shape[0] ({x.shape[0]})"
        )

    # Validate that arrays contain valid numeric values
    if not np.all(np.isfinite(G)):
        raise ValueError("G contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values (NaN or Inf)")

    return G @ x