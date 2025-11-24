"""
Temperature drift compensation algorithms for RRAM devices.

This module implements algorithms to compensate for temperature-induced drift
in RRAM conductance values, which is a critical reliability concern.
"""
import numpy as np
from typing import Tuple, Dict, Optional
from collections import deque
import warnings


def temperature_coefficient_model(conductance: np.ndarray, 
                                temp_ref: float = 300.0,  # Reference temperature in Kelvin
                                temp_op: float = 300.0,   # Operating temperature in Kelvin
                                temp_coeff: float = 0.002) -> np.ndarray:
    """
    Apply temperature coefficient model to adjust conductance for temperature drift.
    
    The temperature coefficient model assumes conductance changes according to:
    G(T) = G_ref * [1 + α * (T - T_ref)]
    
    where α is the temperature coefficient.
    
    Args:
        conductance: Original conductance values
        temp_ref: Reference temperature (Kelvin) - default: 300K (room temperature)
        temp_op: Operating temperature (Kelvin)
        temp_coeff: Temperature coefficient (typically 0.001-0.005 for RRAM)
        
    Returns:
        Adjusted conductance values accounting for temperature effects
    """
    if temp_coeff < -0.1 or temp_coeff > 0.1:
        warnings.warn(f"Temperature coefficient {temp_coeff} seems unusually large")
    
    # Apply temperature coefficient model
    temp_factor = 1.0 + temp_coeff * (temp_op - temp_ref)
    adjusted_conductance = conductance * temp_factor
    
    # Ensure non-negative conductance values
    adjusted_conductance = np.maximum(adjusted_conductance, 0)
    
    return adjusted_conductance


def arrhenius_drift_model(conductance: np.ndarray,
                         initial_time: float,
                         current_time: float,
                         temp: float,
                         activation_energy: float = 0.5,  # Activation energy in eV
                         attempt_freq: float = 1e12) -> np.ndarray:
    """
    Apply Arrhenius model for time-dependent conductance drift.
    
    This model describes the thermally activated drift of resistance states.
    The conductance change follows: ΔG = ΔG_0 * exp(-t/τ), where τ is the
    characteristic time constant that depends on temperature.
    
    Args:
        conductance: Original conductance values
        initial_time: Time when conductance was set (arbitrary units)
        current_time: Current time (arbitrary units)
        temp: Current temperature in Kelvin
        activation_energy: Activation energy for drift process (eV)
        attempt_freq: Attempt frequency (Hz)
        
    Returns:
        Conductance adjusted for time-dependent drift
    """
    # Boltzmann constant in eV/K
    k_B = 8.617e-5  # eV/K
    
    # Calculate characteristic time constant
    # τ = 1/f0 * exp(Ea/kT) where f0 is attempt frequency
    char_time = (1.0 / attempt_freq) * np.exp(activation_energy / (k_B * temp))
    
    # Time elapsed
    time_elapsed = current_time - initial_time
    
    # Calculate drift factor - assuming exponential relaxation
    drift_factor = np.exp(-time_elapsed / char_time)
    
    # Apply drift (this is a simplified model - in practice, drift might be in both directions)
    # For this model, we'll assume conductance relaxes toward a more stable value
    # Let's assume the stable conductance is slightly lower
    stable_factor = 0.98  # Assume 2% relaxation toward stable state
    target_conductance = conductance * stable_factor
    
    # Apply relaxation
    adjusted_conductance = target_conductance + (conductance - target_conductance) * drift_factor
    
    return adjusted_conductance


class TemperatureDriftCompensator:
    """
    A class that implements temperature drift compensation for RRAM devices.

    This class combines multiple compensation techniques to maintain accuracy
    under varying temperature conditions. Uses bounded history buffers to prevent
    memory leaks during long-term operation.
    """

    def __init__(self,
                 temp_coeff: float = 0.002,
                 activation_energy: float = 0.5,
                 calibration_interval: float = 3600.0,  # 1 hour in seconds
                 temp_tolerance: float = 5.0):  # 5K temperature tolerance
        """
        Initialize the temperature drift compensator.

        Args:
            temp_coeff: Temperature coefficient for instantaneous compensation
            activation_energy: Activation energy for time-dependent drift
            calibration_interval: How often to perform full calibration (seconds)
            temp_tolerance: How much temperature change to tolerate before compensation (K)
        """
        self.temp_coeff = temp_coeff
        self.activation_energy = activation_energy
        self.calibration_interval = calibration_interval
        self.temp_tolerance = temp_tolerance

        # Track calibration status
        self.last_calibration_time = 0.0
        self.last_known_temp = 300.0  # Kelvin
        self.baseline_conductance = None
        self.temperature_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        
    def compensate_conductance(self, 
                              conductance: np.ndarray,
                              current_time: float,
                              current_temp: float,
                              set_time: Optional[float] = None) -> np.ndarray:
        """
        Compensate conductance for temperature and time-dependent drift.
        
        Args:
            conductance: Measured conductance values
            current_time: Current time in seconds
            current_temp: Current temperature in Kelvin
            set_time: Time when conductance was initially set (if known)
            
        Returns:
            Temperature-compensated conductance values
        """
        if set_time is None:
            # If we don't know when it was set, assume it was recently
            set_time = current_time - 1.0  # 1 second ago
        
        # Apply temperature coefficient compensation for current temperature
        temp_adjusted = temperature_coefficient_model(
            conductance, 
            temp_op=current_temp,
            temp_coeff=self.temp_coeff
        )
        
        # Apply time-dependent drift compensation using Arrhenius model
        # This assumes the conductance was set at set_time and has been drifting since
        if current_temp > 0:  # Only apply if we have a valid temperature
            drift_adjusted = arrhenius_drift_model(
                temp_adjusted,
                set_time,
                current_time,
                current_temp,
                self.activation_energy
            )
        else:
            drift_adjusted = temp_adjusted
        
        return drift_adjusted
    
    def calibrate_if_needed(self, 
                           conductance: np.ndarray, 
                           current_time: float,
                           current_temp: float) -> Tuple[np.ndarray, bool]:
        """
        Perform calibration if needed based on time or temperature changes.
        
        Args:
            conductance: Current conductance values
            current_time: Current time in seconds
            current_temp: Current temperature in Kelvin
            
        Returns:
            Tuple of (compensated_conductance, calibration_performed)
        """
        # Check if we need calibration based on time
        time_for_calibration = (current_time - self.last_calibration_time) >= self.calibration_interval
        
        # Check if we need calibration based on temperature change
        temp_change = abs(current_temp - self.last_known_temp)
        temp_for_calibration = temp_change >= self.temp_tolerance
        
        calibration_needed = time_for_calibration or temp_for_calibration
        
        if calibration_needed:
            # Store current state as baseline for future compensation
            self.baseline_conductance = conductance.copy()
            self.last_calibration_time = current_time
            self.last_known_temp = current_temp

            # Also add to history for tracking (automatically bounded by deque maxlen)
            self.temperature_history.append((current_time, current_temp))

            return conductance, True
        else:
            # Apply compensation based on history
            compensated = self.compensate_conductance(
                conductance, current_time, current_temp
            )
            return compensated, False
    
    def predict_drift(self, 
                     conductance: np.ndarray,
                     future_time: float,
                     current_time: float,
                     current_temp: float) -> np.ndarray:
        """
        Predict how conductance will drift in the future.
        
        Args:
            conductance: Current conductance values
            future_time: Time in the future to predict for
            current_time: Current time
            current_temp: Current temperature in Kelvin
            
        Returns:
            Predicted conductance at future_time
        """
        # Use the drift model to predict future state
        predicted = arrhenius_drift_model(
            conductance,
            current_time,
            future_time,
            current_temp,
            self.activation_energy
        )
        
        # Also account for temperature changes if known
        return predicted


def apply_temperature_compensation_to_rram_matrix(
    matrix: np.ndarray,
    temp_current: float = 300.0,  # Current temperature in Kelvin
    temp_initial: float = 300.0,   # Temperature when programmed
    time_elapsed: float = 3600.0,  # Time since programming in seconds (1 hour)
    temp_coeff: float = 0.002,
    activation_energy: float = 0.5
) -> np.ndarray:
    """
    Apply temperature compensation to an entire RRAM matrix.
    
    This function combines both instantaneous temperature effects and
    time-dependent drift to adjust an RRAM conductance matrix.
    
    Args:
        matrix: RRAM conductance matrix to compensate
        temp_current: Current temperature (K)
        temp_initial: Programming temperature (K)
        time_elapsed: Time since programming (seconds)
        temp_coeff: Temperature coefficient
        activation_energy: Activation energy for drift
        
    Returns:
        Temperature-compensated RRAM matrix
    """
    # First apply instantaneous temperature compensation
    temp_compensated = temperature_coefficient_model(
        matrix, temp_ref=temp_initial, temp_op=temp_current, temp_coeff=temp_coeff
    )
    
    # Then apply time-dependent drift compensation
    current_time = 0.0  # Reference
    set_time = -time_elapsed  # Time when it was set
    
    drift_compensated = arrhenius_drift_model(
        temp_compensated,
        set_time,
        current_time,  # Currently
        temp_current,
        activation_energy
    )
    
    return drift_compensated


def simulate_temperature_drift_over_time(
    initial_matrix: np.ndarray,
    time_points: np.ndarray,
    temperature_profile: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Simulate how an RRAM matrix drifts over time with varying temperature.
    
    Args:
        initial_matrix: Original RRAM matrix
        time_points: Time points at which to evaluate drift (seconds)
        temperature_profile: Temperature at each time point (Kelvin)
        
    Returns:
        Dictionary containing drift information over time
    """
    if len(time_points) != len(temperature_profile):
        raise ValueError("time_points and temperature_profile must have same length")
    
    # Track the evolution of the matrix over time
    matrices_over_time = []
    
    # Start with the initial matrix (compensated to time 0, temp 300K)
    current_matrix = initial_matrix.copy()
    current_time = 0.0
    
    for i, (time, temp) in enumerate(zip(time_points, temperature_profile)):
        # Since we're simulating drift from initial state, we need to calculate
        # how the matrix would evolve from the initial state to each time point
        
        # For this simulation, we'll apply drift from initial time to current time point
        drifted_matrix = arrhenius_drift_model(
            initial_matrix,  # Start from initial
            0.0,  # Started at time 0
            time,  # Currently at this time
            temp,  # At this temperature
            activation_energy=0.5  # Default activation energy
        )
        
        # Apply instantaneous temperature adjustment
        temp_adjusted = temperature_coefficient_model(
            drifted_matrix,
            temp_ref=300.0,  # Reference temperature
            temp_op=temp,    # Current temperature
            temp_coeff=0.002  # Default temp coefficient
        )
        
        matrices_over_time.append(temp_adjusted)
    
    return {
        'time_points': time_points,
        'temperature_profile': temperature_profile,
        'matrices': matrices_over_time
    }


# Example usage and testing functions
def test_temperature_compensation():
    """Test the temperature compensation algorithms."""
    np.random.seed(42)
    
    print("Testing Temperature Compensation Algorithms")
    print("=" * 50)
    
    # Create a sample RRAM matrix
    original_matrix = np.random.rand(4, 4) * 30e-6 + 1e-6  # 1-31 μS range
    print(f"Original matrix range: {original_matrix.min():.2e} to {original_matrix.max():.2e} S")
    
    # Test temperature coefficient model
    print("\n1. Temperature Coefficient Model:")
    temp_adjusted = temperature_coefficient_model(original_matrix, 
                                                 temp_ref=300, temp_op=330)  # 300K to 330K
    print(f"   After temperature change (300K→330K): {temp_adjusted.min():.2e} to {temp_adjusted.max():.2e} S")
    
    # Test Arrhenius drift model
    print("\n2. Arrhenius Drift Model:")
    t0 = 0.0
    t1 = 3600.0  # 1 hour later
    temp = 300.0  # Constant temperature
    drift_adjusted = arrhenius_drift_model(original_matrix, t0, t1, temp)
    print(f"   After 1 hour drift: {drift_adjusted.min():.2e} to {drift_adjusted.max():.2e} S")
    
    # Test the compensator class
    print("\n3. Temperature Drift Compensator:")
    compensator = TemperatureDriftCompensator()
    
    # Simulate a matrix measurement
    measured_matrix = original_matrix.copy()
    drift_compensated, calibrated = compensator.calibrate_if_needed(
        measured_matrix, current_time=3600, current_temp=310
    )
    print(f"   Compensated matrix range: {drift_compensated.min():.2e} to {drift_compensated.max():.2e} S")
    print(f"   Calibration performed: {calibrated}")
    
    # Test full compensation function
    print("\n4. Full Temperature Compensation:")
    fully_compensated = apply_temperature_compensation_to_rram_matrix(
        original_matrix,
        temp_current=320,
        temp_initial=300,
        time_elapsed=7200  # 2 hours
    )
    print(f"   Fully compensated range: {fully_compensated.min():.2e} to {fully_compensated.max():.2e} S")
    
    print("\nTemperature compensation tests completed!")


if __name__ == "__main__":
    test_temperature_compensation()