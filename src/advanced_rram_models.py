"""
Advanced RRAM models with temperature effects, aging, and physics-based modeling.

This module provides more sophisticated RRAM models that include:
- Temperature-dependent conductance
- Time-dependent drift
- Conductive filament dynamics
- Material-specific characteristics
- Aging effects
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time
from enum import Enum
from collections import deque
import threading


class RRAMMaterialType(Enum):
    """Types of RRAM materials with different characteristics."""
    HFO2 = "HfO2"  # Hafnium oxide
    TAOX = "TaOx"  # Tantalum oxide
    TIO2 = "TiO2"  # Titanium dioxide
    NIOX = "NiOx"  # Nickel oxide
    COOX = "CoOx"  # Cobalt oxide


@dataclass
class RRAMParameters:
    """Parameters for RRAM device modeling."""
    # Basic electrical parameters
    low_resistance: float = 100.0      # R_low in ohms
    high_resistance: float = 10000.0   # R_high in ohms
    init_conductance: float = 1e-4     # Initial conductance in siemens
    max_conductance: float = 1e-3      # Maximum conductance in siemens
    
    # Temperature parameters
    alpha: float = 0.004               # Temperature coefficient
    reference_temp: float = 300.0      # Reference temperature in Kelvin
    thermal_resistance: float = 50.0   # Thermal resistance in K/W
    
    # Aging and drift parameters
    drift_coefficient: float = 1e-4    # Conductance drift coefficient
    aging_factor: float = 1e-6         # Aging factor for long-term drift
    cycle_to_cycle_var: float = 0.05   # Cycle-to-cycle variation
    
    # Physics parameters
    filament_core_radius: float = 2.5e-9  # Core radius in meters
    filament_shell_width: float = 1.0e-9  # Shell width in meters
    oxygen_vacancy_density: float = 1e25   # m^-3
    migration_barrier: float = 0.7         # eV
    attempt_freq: float = 1e13             # Hz
    
    # Material-specific adjustments
    material: RRAMMaterialType = RRAMMaterialType.HFO2


class AdvancedRRAMModel:
    """
    Advanced RRAM model with temperature effects, aging, and physics-based modeling.
    """

    # Maximum history size to prevent memory leaks
    MAX_HISTORY_SIZE = 1000

    def __init__(self, params: Optional[RRAMParameters] = None):
        """
        Initialize advanced RRAM model with physics-based parameters.

        Args:
            params: RRAMParameters object with model parameters
        """
        self.params = params or RRAMParameters()
        self.temperature = self.params.reference_temp  # Current temperature in Kelvin
        self.time = 0.0  # Elapsed time in seconds
        self.cycle_count = 0  # Number of switching cycles
        # Use deque with maxlen to prevent unbounded memory growth
        self.conductance_history = deque(maxlen=self.MAX_HISTORY_SIZE)
        self.current_conductance = self.params.init_conductance
        
    def update_temperature(self, new_temp: float) -> None:
        """Update the temperature of the RRAM device."""
        self.temperature = new_temp
        
    def update_time(self, elapsed_time: float) -> None:
        """Update the elapsed time for aging calculations."""
        self.time += elapsed_time
        
    def get_temperature_dependent_conductance(self, base_conductance: float) -> float:
        """Calculate conductance adjusted for temperature."""
        temp_factor = 1.0 + self.params.alpha * (self.temperature - self.params.reference_temp)
        return base_conductance * temp_factor
    
    def get_aging_effect(self) -> float:
        """Calculate conductance drift due to aging."""
        return self.params.aging_factor * self.time * self.cycle_count
    
    def get_cycle_to_cycle_variation(self) -> float:
        """Add cycle-to-cycle variation."""
        return np.random.normal(0, self.params.cycle_to_cycle_var)
    
    def simulate_conductive_filament_dynamics(
        self, 
        voltage: float, 
        dt: float
    ) -> float:
        """
        Simulate conductive filament dynamics based on voltage and time.
        
        Args:
            voltage: Applied voltage across the device
            dt: Time step for the simulation
            
        Returns:
            Change in conductance due to filament dynamics
        """
        # Simplified model based on oxygen vacancy migration
        # J = D * n * q * mu * E / kT (current density)
        # where D is diffusion coefficient, n is vacancy density, etc.
        
        # For simplicity, use an exponential model
        k_boltzmann = 8.617e-5  # eV/K
        field_factor = abs(voltage) / 10.0  # Normalize voltage to field effect
        
        # Arrhenius equation for migration rate
        migration_rate = self.params.attempt_freq * np.exp(
            -self.params.migration_barrier / (k_boltzmann * self.temperature)
        )
        
        # Change in conductance based on voltage polarity and migration
        if voltage > 0:  # Set operation (increase conductance)
            delta_g = 1e-6 * field_factor * migration_rate * dt
        elif voltage < 0:  # Reset operation (decrease conductance)
            delta_g = -1e-6 * field_factor * migration_rate * dt
        else:  # No voltage, possible drift due to thermal effects
            delta_g = np.random.normal(0, 1e-8) * dt
        
        return delta_g
    
    def update_conductance(
        self, 
        target_conductance: float, 
        voltage: float = 0.0, 
        dt: float = 1e-6
    ) -> float:
        """
        Update the conductance based on physics models.
        
        Args:
            target_conductance: Desired conductance value
            voltage: Applied voltage
            dt: Time step
            
        Returns:
            Updated conductance value
        """
        # Apply temperature effects
        temp_adjusted = self.get_temperature_dependent_conductance(target_conductance)
        
        # Apply aging effects
        aging_drift = self.get_aging_effect()
        age_adjusted = temp_adjusted + aging_drift
        
        # Add cycle-to-cycle variation
        ctc_variation = self.get_cycle_to_cycle_variation()
        ctc_adjusted = age_adjusted * (1.0 + ctc_variation)
        
        # Simulate filament dynamics (simplified)
        filament_change = self.simulate_conductive_filament_dynamics(voltage, dt)
        new_conductance = ctc_adjusted + filament_change
        
        # Apply physical limits
        new_conductance = np.clip(
            new_conductance, 
            1.0 / self.params.high_resistance,  # Min conductance
            1.0 / self.params.low_resistance    # Max conductance
        )
        
        # Update history and current value
        self.current_conductance = new_conductance
        self.conductance_history.append(new_conductance)
        self.cycle_count += 1
        
        return new_conductance
    
    def get_current_with_noise(self, voltage: float) -> float:
        """
        Calculate current through the device including thermal and shot noise.
        
        Args:
            voltage: Applied voltage
            
        Returns:
            Current through the device with noise
        """
        # Basic current calculation
        current = self.current_conductance * voltage
        
        # Add thermal noise (Johnson-Nyquist noise)
        k_boltzmann = 1.38e-23  # J/K
        bandwidth = 1e9  # 1 GHz bandwidth
        thermal_noise = np.sqrt(4 * k_boltzmann * self.temperature * 
                               self.current_conductance * bandwidth)
        
        # Add shot noise
        q_electron = 1.6e-19  # C
        shot_noise = np.sqrt(2 * q_electron * abs(current) * bandwidth)
        
        # Combine noise sources
        total_noise = np.sqrt(thermal_noise**2 + shot_noise**2)
        noise = np.random.normal(0, total_noise)
        
        return current + noise


class MaterialSpecificRRAMModel(AdvancedRRAMModel):
    """
    Material-specific RRAM model with different characteristics for different materials.
    """
    
    def __init__(self, material_type: RRAMMaterialType = RRAMMaterialType.HFO2):
        """
        Initialize with material-specific parameters.
        
        Args:
            material_type: Type of RRAM material to model
        """
        params = self._get_material_parameters(material_type)
        super().__init__(params)
        
    def _get_material_parameters(self, material_type: RRAMMaterialType) -> RRAMParameters:
        """
        Get material-specific parameters for different RRAM types.
        
        Args:
            material_type: Type of RRAM material
            
        Returns:
            RRAMParameters object with material-specific values
        """
        if material_type == RRAMMaterialType.HFO2:
            # HfO2: Binary switching, good endurance
            return RRAMParameters(
                low_resistance=50.0,
                high_resistance=50000.0,
                init_conductance=5e-5,
                max_conductance=2e-3,
                alpha=0.0035,  # Temperature coefficient
                drift_coefficient=8e-5,
                aging_factor=8e-7,
                cycle_to_cycle_var=0.04,
                filament_core_radius=2.2e-9,
                oxygen_vacancy_density=8e24,
                migration_barrier=0.65,
                material=material_type
            )
        elif material_type == RRAMMaterialType.TAOX:
            # TaOx: Analog switching, good linearity
            return RRAMParameters(
                low_resistance=100.0,
                high_resistance=100000.0,
                init_conductance=1e-4,
                max_conductance=1e-3,
                alpha=0.0045,
                drift_coefficient=1.2e-4,
                aging_factor=1.2e-6,
                cycle_to_cycle_var=0.06,
                filament_core_radius=2.5e-9,
                oxygen_vacancy_density=1.2e25,
                migration_barrier=0.75,
                material=material_type
            )
        elif material_type == RRAMMaterialType.TIO2:
            # TiO2: Slow switching, good retention
            return RRAMParameters(
                low_resistance=200.0,
                high_resistance=20000.0,
                init_conductance=2e-4,
                max_conductance=5e-4,
                alpha=0.0025,
                drift_coefficient=6e-5,
                aging_factor=6e-7,
                cycle_to_cycle_var=0.03,
                filament_core_radius=3.0e-9,
                oxygen_vacancy_density=6e24,
                migration_barrier=0.8,
                material=material_type
            )
        elif material_type == RRAMMaterialType.NIOX:
            # NiOx: Electrochemical metallization (ECM)
            return RRAMParameters(
                low_resistance=80.0,
                high_resistance=80000.0,
                init_conductance=8e-5,
                max_conductance=8e-4,
                alpha=0.005,
                drift_coefficient=1.5e-4,
                aging_factor=1.5e-6,
                cycle_to_cycle_var=0.07,
                filament_core_radius=1.8e-9,
                oxygen_vacancy_density=1.5e25,
                migration_barrier=0.55,
                material=material_type
            )
        elif material_type == RRAMMaterialType.COOX:
            # CoOx: Mixed switching mechanism
            return RRAMParameters(
                low_resistance=150.0,
                high_resistance=150000.0,
                init_conductance=1.5e-4,
                max_conductance=1.5e-3,
                alpha=0.004,
                drift_coefficient=1e-4,
                aging_factor=1e-6,
                cycle_to_cycle_var=0.05,
                filament_core_radius=2.7e-9,
                oxygen_vacancy_density=9e24,
                migration_barrier=0.72,
                material=material_type
            )
        else:
            # Default to HfO2 parameters
            return RRAMParameters(material=material_type)


class TemporalRRAMModel(AdvancedRRAMModel):
    """
    RRAM model with explicit time-dependent effects including retention and drift.

    This class is thread-safe for concurrent access to time-dependent state.
    """

    # Maximum programming history size to prevent memory leaks
    MAX_PROGRAMMING_HISTORY_SIZE = 1000

    def __init__(self, params: Optional[RRAMParameters] = None):
        super().__init__(params)
        self.last_update_time = time.time()
        self.retention_time = 0.0
        self.time_since_programming = 0.0
        # Use deque with maxlen to prevent unbounded memory growth
        self.programming_history = deque(maxlen=self.MAX_PROGRAMMING_HISTORY_SIZE)
        # Thread lock for protecting shared mutable state
        self._lock = threading.Lock()
        
    def program_conductance(
        self,
        target_conductance: float,
        voltage_pulse: float,
        pulse_duration: float
    ) -> bool:
        """
        Program the RRAM to a target conductance with voltage pulse.

        This method is thread-safe.

        Args:
            target_conductance: Target conductance value
            voltage_pulse: Programming voltage
            pulse_duration: Duration of the pulse

        Returns:
            True if programming successful, False otherwise
        """
        with self._lock:
            try:
                # Update time since programming
                current_time = time.time()
                elapsed = current_time - self.last_update_time
                self.time_since_programming += elapsed
                self.last_update_time = current_time

                # Apply the programming pulse
                new_conductance = self.update_conductance(
                    target_conductance,
                    voltage_pulse,
                    pulse_duration
                )

                # Record programming event
                self.programming_history.append({
                    'time': self.time_since_programming,
                    'target': target_conductance,
                    'actual': new_conductance,
                    'voltage': voltage_pulse,
                    'duration': pulse_duration
                })

                return True
            except Exception as e:
                # Log the exception for debugging
                import logging
                logging.getLogger(self.__class__.__name__).error(f"Programming failed: {e}")
                return False
    
    def get_retention_affected_conductance(self) -> float:
        """
        Get conductance affected by retention loss over time.

        This method is thread-safe.

        Returns:
            Conductance adjusted for retention loss
        """
        with self._lock:
            # Simple retention model: exponential decay based on time since programming
            retention_factor = np.exp(-self.time_since_programming * 1e-6)  # Adjust time constant
            return self.current_conductance * retention_factor

    def simulate_drift(self, time_elapsed: float) -> float:
        """
        Simulate time-dependent drift in conductance.

        This method is thread-safe.

        Args:
            time_elapsed: Time elapsed since last measurement

        Returns:
            Change in conductance due to drift
        """
        with self._lock:
            # Time-dependent drift following power law: drift = A * t^gamma
            drift_coeff = self.params.drift_coefficient
            gamma = 0.1  # Typical for RRAM devices

            drift = drift_coeff * (time_elapsed ** gamma)
            return drift


class TDDBModel(AdvancedRRAMModel):
    """
    Time-Dependent Dielectric Breakdown model for RRAM reliability.
    """
    
    def __init__(self, params: Optional[RRAMParameters] = None):
        super().__init__(params)
        self.tddb_stress = 0.0
        self.failure_probability = 0.0
        
    def apply_stress(self, voltage: float, time: float) -> bool:
        """
        Apply electrical stress and update TDDB parameters.
        
        Args:
            voltage: Applied voltage
            time: Duration of stress application
            
        Returns:
            True if device still functional, False if failed
        """
        # TDDB model: increase stress with voltage and time
        voltage_factor = np.exp(abs(voltage) / 0.5)  # E-model
        self.tddb_stress += voltage_factor * time
        
        # Calculate failure probability using Weibull distribution
        self.failure_probability = 1.0 - np.exp(-(self.tddb_stress / 1e6) ** 2)
        
        # Determine if the device has failed
        return np.random.random() > self.failure_probability


# Factory function to create appropriate model based on requirements
def create_advanced_rram_model(
    model_type: str = "advanced", 
    material: RRAMMaterialType = RRAMMaterialType.HFO2,
    params: Optional[RRAMParameters] = None
) -> AdvancedRRAMModel:
    """
    Factory function to create advanced RRAM models.
    
    Args:
        model_type: Type of model ("advanced", "material_specific", "temporal", "tddb")
        material: Material type for material-specific model
        params: Optional custom parameters
        
    Returns:
        Advanced RRAM model instance
    """
    if model_type == "material_specific":
        return MaterialSpecificRRAMModel(material_type=material)
    elif model_type == "temporal":
        return TemporalRRAMModel(params=params)
    elif model_type == "tddb":
        return TDDBModel(params=params)
    else:
        return AdvancedRRAMModel(params=params)


class RRAMNetworkModel:
    """
    Model of RRAM crossbar network with multiple devices and interconnect effects.
    """
    
    def __init__(
        self, 
        rows: int, 
        cols: int, 
        row_resistance: float = 1.0, 
        col_resistance: float = 1.0
    ):
        """
        Initialize RRAM crossbar network model.
        
        Args:
            rows: Number of rows in crossbar
            cols: Number of columns in crossbar
            row_resistance: Resistance of row lines
            col_resistance: Resistance of column lines
        """
        self.rows = rows
        self.cols = cols
        self.row_resistance = row_resistance
        self.col_resistance = col_resistance
        
        # Initialize network with individual RRAM devices
        self.devices = np.array([
            [create_advanced_rram_model("material_specific", 
                                      np.random.choice(list(RRAMMaterialType))) 
             for _ in range(cols)] 
            for _ in range(rows)
        ])
        
        # Track interconnect voltages
        self.row_voltages = np.zeros(rows)
        self.col_voltages = np.zeros(cols)
    
    def apply_voltage_pattern(self, voltage_matrix: np.ndarray) -> np.ndarray:
        """
        Apply voltage pattern across the crossbar and calculate resulting currents.
        
        Args:
            voltage_matrix: Matrix of applied voltages (rows x cols)
            
        Returns:
            Matrix of resulting currents
        """
        current_matrix = np.zeros_like(voltage_matrix)
        
        for i in range(self.rows):
            for j in range(self.cols):
                # Apply voltage and get current including device non-idealities
                device = self.devices[i][j]
                voltage = voltage_matrix[i, j]
                current = device.get_current_with_noise(voltage)
                current_matrix[i, j] = current
        
        return current_matrix
    
    def matrix_vector_multiply(self, vector: np.ndarray) -> np.ndarray:
        """
        Perform matrix-vector multiplication using crossbar network.
        
        Args:
            vector: Input vector
            
        Returns:
            Result of matrix-vector multiplication
        """
        if len(vector) != self.cols:
            raise ValueError(f"Vector length {len(vector)} doesn't match columns {self.cols}")
        
        # Apply input vector as voltages to columns
        for j in range(self.cols):
            self.col_voltages[j] = vector[j]
        
        # Calculate output by applying conductance matrix
        output = np.zeros(self.rows)
        
        for i in range(self.rows):
            for j in range(self.cols):
                # Current through device i,j
                device = self.devices[i][j]
                voltage = self.col_voltages[j] - 0  # Assuming ground for row
                current = device.current_conductance * voltage
                
                # Add to output (with interconnect effects simplified)
                output[i] += current
        
        return output