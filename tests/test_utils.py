"""Tests for the utility functions module."""
import numpy as np
import pytest
from src.utils import (
    validate_matrix_vector_inputs,
    validate_matrix_inputs,
    validate_vector_inputs,
    validate_parameter,
    quantize,
    matrix_vector_multiply
)


def test_validate_matrix_vector_inputs_valid():
    """Test that valid inputs pass validation."""
    G = np.random.rand(3, 3)
    b = np.random.rand(3)
    validate_matrix_vector_inputs(G, b, "test_func")


def test_validate_matrix_vector_inputs_invalid_type():
    """Test validation with invalid input types."""
    with pytest.raises(TypeError, match="G must be a numpy array"):
        validate_matrix_vector_inputs([[1, 2], [3, 4]], np.array([1, 2]), "test_func")

    with pytest.raises(TypeError, match="b must be a numpy array"):
        validate_matrix_vector_inputs(np.array([[1, 2], [3, 4]]), [1, 2], "test_func")


def test_validate_matrix_vector_inputs_invalid_dimensions():
    """Test validation with invalid dimensions."""
    # Test non-2D matrix
    with pytest.raises(ValueError, match="G must be a 2D matrix"):
        validate_matrix_vector_inputs(np.array([1, 2, 3]), np.array([1, 2, 3]), "test_func")

    # Test non-1D vector
    with pytest.raises(ValueError, match="b must be a 1D vector"):
        validate_matrix_vector_inputs(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]), "test_func")

    # Test non-square matrix
    with pytest.raises(ValueError, match="G must be square"):
        validate_matrix_vector_inputs(np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2]), "test_func")

    # Test dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_matrix_vector_inputs(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3]), "test_func")


def test_validate_matrix_vector_inputs_non_finite():
    """Test validation with non-finite values."""
    G = np.array([[1, 2], [3, np.inf]])
    b = np.array([1, 2])
    with pytest.raises(ValueError, match="G contains non-finite values"):
        validate_matrix_vector_inputs(G, b, "test_func")

    G = np.array([[1, 2], [3, 4]])
    b = np.array([1, np.nan])
    with pytest.raises(ValueError, match="b contains non-finite values"):
        validate_matrix_vector_inputs(G, b, "test_func")


def test_validate_matrix_inputs_valid():
    """Test that valid matrix inputs pass validation."""
    G = np.random.rand(3, 3)
    validate_matrix_inputs(G, func_name="test_func")


def test_validate_matrix_inputs_invalid_type():
    """Test validation with invalid matrix type."""
    with pytest.raises(TypeError, match="G must be a numpy array"):
        validate_matrix_inputs([[1, 2], [3, 4]], func_name="test_func")


def test_validate_matrix_inputs_invalid_dimensions():
    """Test validation with invalid matrix dimensions."""
    with pytest.raises(ValueError, match="G must be a 2D matrix"):
        validate_matrix_inputs(np.array([1, 2, 3]), func_name="test_func")


def test_validate_matrix_inputs_non_finite():
    """Test validation with non-finite matrix values."""
    G = np.array([[1, 2], [3, np.inf]])
    with pytest.raises(ValueError, match="G contains non-finite values"):
        validate_matrix_inputs(G, func_name="test_func")


def test_validate_vector_inputs_valid():
    """Test that valid vector inputs pass validation."""
    x = np.random.rand(3)
    validate_vector_inputs(x, func_name="test_func")


def test_validate_vector_inputs_invalid_type():
    """Test validation with invalid vector type."""
    with pytest.raises(TypeError, match="x must be a numpy array"):
        validate_vector_inputs([1, 2, 3], func_name="test_func")


def test_validate_vector_inputs_invalid_dimensions():
    """Test validation with invalid vector dimensions."""
    with pytest.raises(ValueError, match="x must be a 1D vector"):
        validate_vector_inputs(np.array([[1], [2]]), func_name="test_func")


def test_validate_vector_inputs_non_finite():
    """Test validation with non-finite vector values."""
    x = np.array([1, 2, np.nan])
    with pytest.raises(ValueError, match="x contains non-finite values"):
        validate_vector_inputs(x, func_name="test_func")


def test_validate_parameter_valid():
    """Test validation of valid parameters."""
    # Test valid integer parameter
    validate_parameter(5, "test_int", min_value=1, integer=True, func_name="test_func")
    
    # Test valid float parameter
    validate_parameter(2.5, "test_float", min_value=0, func_name="test_func")


def test_validate_parameter_invalid_type():
    """Test validation with invalid parameter types."""
    with pytest.raises(TypeError, match="test_int must be an integer"):
        validate_parameter(5.5, "test_int", integer=True, func_name="test_func")

    with pytest.raises(TypeError, match="test_num must be numeric"):
        validate_parameter("not_a_number", "test_num", func_name="test_func")


def test_validate_parameter_bounds():
    """Test parameter validation with bounds."""
    with pytest.raises(ValueError, match="test_param must be >= 1"):
        validate_parameter(0, "test_param", min_value=1, func_name="test_func")

    with pytest.raises(ValueError, match="test_param must be <= 10"):
        validate_parameter(15, "test_param", max_value=10, func_name="test_func")


def test_validate_parameter_positive():
    """Test validation of positive parameter constraint."""
    with pytest.raises(ValueError, match="test_param must be positive"):
        validate_parameter(-1, "test_param", positive=True, func_name="test_func")


def test_quantize_basic():
    """Test basic quantization functionality."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    quantized = quantize(matrix, 3)
    
    # Check that output is same shape as input
    assert quantized.shape == matrix.shape
    # Check that values are finite
    assert np.all(np.isfinite(quantized))


def test_quantize_bits_validation():
    """Test quantization with invalid bits parameter."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    with pytest.raises(ValueError, match="Number of bits must be positive"):
        quantize(matrix, 0)


def test_quantize_edge_cases():
    """Test quantization edge cases."""
    # Empty matrix
    empty_matrix = np.array([]).reshape(0, 0)
    result = quantize(empty_matrix, 3)
    assert result.shape == empty_matrix.shape

    # Zero matrix
    zero_matrix = np.zeros((2, 2))
    result = quantize(zero_matrix, 3)
    assert np.array_equal(result, zero_matrix)

    # Matrix with all same values
    same_val_matrix = np.ones((2, 2))
    result = quantize(same_val_matrix, 1)
    assert result.shape == same_val_matrix.shape


def test_matrix_vector_multiply_valid():
    """Test valid matrix-vector multiplication."""
    G = np.array([[2.0, 1.0], [1.0, 3.0]])
    x = np.array([1.0, 2.0])
    result = matrix_vector_multiply(G, x)
    
    expected = G @ x
    np.testing.assert_array_almost_equal(result, expected)


def test_matrix_vector_multiply_validation():
    """Test matrix-vector multiplication with invalid inputs."""
    # Wrong matrix dimensions
    G = np.array([1.0, 2.0])  # 1D array instead of 2D
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        matrix_vector_multiply(G, x)

    # Wrong vector dimensions
    G = np.array([[2.0, 1.0], [1.0, 3.0]])
    x = np.array([[1.0], [2.0]])  # 2D array instead of 1D
    with pytest.raises(ValueError):
        matrix_vector_multiply(G, x)

    # Incompatible dimensions
    G = np.array([[2.0, 1.0, 3.0], [1.0, 3.0, 2.0]])  # 2x3 matrix
    x = np.array([1.0, 2.0])  # 2-element vector
    with pytest.raises(ValueError):
        matrix_vector_multiply(G, x)

    # Non-finite values
    G = np.array([[2.0, np.inf], [1.0, 3.0]])
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        matrix_vector_multiply(G, x)