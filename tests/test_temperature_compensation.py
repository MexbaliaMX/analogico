"""Tests for temperature compensation module."""
import numpy as np
import pytest
from src.temperature_compensation import (
    temperature_coefficient_model,
    arrhenius_drift_model,
    TemperatureDriftCompensator,
    apply_temperature_compensation_to_rram_matrix,
    simulate_temperature_drift_over_time
)


def test_temperature_coefficient_model():
    """Test temperature coefficient model."""
    # Basic functionality test
    G = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = temperature_coefficient_model(G, temp_op=320.0, temp_coeff=0.002)

    # Should return same shape
    assert result.shape == G.shape
    # Should have changed values due to temperature effect
    assert not np.array_equal(result, G)  # Values should have changed


def test_temperature_coefficient_model_edge_cases():
    """Test temperature coefficient model edge cases."""
    # Test with identical reference and operating temperatures
    G = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = temperature_coefficient_model(G, temp_ref=300.0, temp_op=300.0, temp_coeff=0.002)
    np.testing.assert_array_almost_equal(result, G)  # Should be unchanged

    # Test with zero temperature coefficient
    result = temperature_coefficient_model(G, temp_coeff=0.0)
    np.testing.assert_array_almost_equal(result, G)  # Should be unchanged

    # Test with negative conductances (should be made positive)
    G_neg = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    result = temperature_coefficient_model(G_neg)
    assert np.all(result >= 0)  # All values should be non-negative


def test_arrhenius_drift_model():
    """Test Arrhenius drift model."""
    # Basic functionality test
    G = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = arrhenius_drift_model(G, 0.0, 3600.0, 300.0, 0.5)  # 1 hour at 300K

    # Should return same shape
    assert result.shape == G.shape


def test_arrhenius_drift_model_edge_cases():
    """Test Arrhenius drift model edge cases."""
    G = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Test with no time elapsed (should be similar to original, with slight relaxation)
    result = arrhenius_drift_model(G, 0.0, 0.0, 300.0, 0.5)
    # Values should be close to original but not identical due to relaxation model
    assert result.shape == G.shape


def test_temperature_drift_compensator():
    """Test TemperatureDriftCompensator class."""
    compensator = TemperatureDriftCompensator()

    # Test initialization
    assert compensator.temp_coeff == 0.002
    assert compensator.activation_energy == 0.5
    assert compensator.calibration_interval == 3600.0
    assert compensator.temp_tolerance == 5.0

    # Test compensate_conductance
    G = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = compensator.compensate_conductance(G, 3600.0, 310.0)
    assert result.shape == G.shape

    # Test calibrate_if_needed
    result, calibrated = compensator.calibrate_if_needed(G, 3600.0, 310.0)
    assert result.shape == G.shape
    # Should be calibrated if enough time has passed
    assert calibrated  # Based on time interval


def test_apply_temperature_compensation_to_rram_matrix():
    """Test apply_temperature_compensation_to_rram_matrix function."""
    original_matrix = np.random.rand(3, 3) * 30e-6 + 1e-6
    result = apply_temperature_compensation_to_rram_matrix(
        original_matrix,
        temp_current=320,
        temp_initial=300,
        time_elapsed=7200,  # 2 hours
    )

    assert result.shape == original_matrix.shape


def test_simulate_temperature_drift_over_time():
    """Test simulate_temperature_drift_over_time function."""
    initial_matrix = np.random.rand(2, 2) * 30e-6 + 1e-6
    time_points = np.array([0, 3600, 7200])  # 0, 1, 2 hours
    temperature_profile = np.array([300, 310, 320])  # Gradually increasing temperature

    result = simulate_temperature_drift_over_time(
        initial_matrix, time_points, temperature_profile
    )

    # Check that the result contains expected keys
    assert 'time_points' in result
    assert 'temperature_profile' in result
    assert 'matrices' in result

    # Check that the number of matrices matches the number of time points
    assert len(result['matrices']) == len(time_points)

    # Check that each matrix has the correct shape
    for matrix in result['matrices']:
        assert matrix.shape == initial_matrix.shape


def test_simulate_temperature_drift_over_time_errors():
    """Test simulate_temperature_drift_over_time error handling."""
    initial_matrix = np.random.rand(2, 2)
    time_points = np.array([0, 3600])
    temperature_profile = np.array([300])  # Different length

    with pytest.raises(ValueError, match="time_points and temperature_profile must have same length"):
        simulate_temperature_drift_over_time(initial_matrix, time_points, temperature_profile)