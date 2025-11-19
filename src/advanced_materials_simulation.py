"""
Advanced Materials Simulation for RRAM Systems.

This module provides enhanced modeling for various RRAM materials
and their specific characteristics, including physics-based models
for different switching mechanisms, temperature effects, and
time-dependent behaviors.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import warnings
from dataclasses import dataclass
import time

# Import from existing modules
try:
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel, RRAMMaterialType
    from .hardware_interface import HardwareInterface
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class SwitchingMechanism(Enum):
    """Types of RRAM switching mechanisms."""
    VCM = "VCM"  # Valence Change Mechanism (OxRAM)
    ECM = "ECM"  # Electrochemical Metallization (CBRAM)
    MIXED = "MIXED"  # Mixed mechanism
    TUNNELING = "TUNNELING"  # Quantum tunneling (PoM)


class TemperatureModel(Enum):
    """Types of temperature models."""
    CONSTANT = "constant"
    THERMAL = "thermal"
    THERMODYNAMIC = "thermodynamic"


@dataclass
class MaterialParameters:
    """Parameters for RRAM materials."""
    # Basic electrical properties
    low_resistance: float = 100.0      # R_low in ohms
    high_resistance: float = 10000.0   # R_high in ohms
    init_conductance: float = 1e-4     # Initial conductance in siemens
    
    # Physical properties
    filament_radius: float = 2.5e-9     # Core radius in meters
    shell_width: float = 1.0e-9         # Shell width in meters
    activation_energy: float = 0.7       # Activation energy in eV
    attempt_freq: float = 1e13          # Attempt frequency in Hz
    
    # Switching properties
    switching_voltage: float = 1.0      # Required switching voltage
    switching_time: float = 1e-8        # Typical switching time in seconds
    switching_mechanism: SwitchingMechanism = SwitchingMechanism.VCM
    
    # Temperature properties
    thermal_conductivity: float = 1.0    # W/(m*K)
    heat_capacity: float = 700.0         # J/(kg*K)
    thermal_expansion: float = 4.7e-6   # 1/K
    temperature_coefficient: float = 0.004  # 1/K
    
    # Aging and reliability properties
    time_constant: float = 1e6           # Time constant for drift
    degradation_rate: float = 1e-12      # Per second at room temp
    retention_time: float = 1e7          # Data retention time in seconds
    
    # Process variation properties
    process_variation: float = 0.1       # Process variation factor
    cycle_to_cycle: float = 0.05         # Cycle-to-cycle variation
    
    # Material-specific properties
    oxygen_vacancy_density: float = 1e25  # m^-3 (for VCM)
    metal_ion_concentration: float = 1e24  # m^-3 (for ECM)


class MaterialSpecificModel:
    """
    Base class for material-specific RRAM models.
    """
    
    def __init__(self, params: MaterialParameters):
        self.params = params
        self.current_conductance = params.init_conductance
        self.temperature = 300.0  # Kelvin
        self.time = 0.0  # Elapsed time in seconds
        self.cycle_count = 0  # Number of switching cycles
        self.switching_history = []
    
    def update_temperature(self, new_temp: float) -> None:
        """Update the temperature of the device."""
        self.temperature = new_temp
    
    def update_time(self, elapsed_time: float) -> None:
        """Update the elapsed time for aging calculations."""
        self.time += elapsed_time
    
    def get_conductance_with_temperature(self, base_conductance: float) -> float:
        """Calculate conductance adjusted for temperature."""
        temp_factor = 1.0 + self.params.temperature_coefficient * (self.temperature - 300.0)
        return base_conductance * temp_factor
    
    def get_aging_effect(self) -> float:
        """Calculate conductance drift due to aging."""
        # Arrhenius equation for temperature acceleration
        k_boltzmann = 8.617e-5  # eV/K
        temp_acceleration = np.exp(
            -self.params.activation_energy * (1.0/self.temperature - 1.0/300.0) / k_boltzmann
        )
        
        # Time-dependent aging
        aging_drift = self.params.degradation_rate * self.time * temp_acceleration
        return aging_drift
    
    def get_process_variation(self) -> float:
        """Add process variation to conductance."""
        return np.random.normal(0, self.params.process_variation)
    
    def get_cycle_to_cycle_variation(self) -> float:
        """Add cycle-to-cycle variation."""
        return np.random.normal(0, self.params.cycle_to_cycle)
    
    def update_conductance(self, target_conductance: float, voltage: float, dt: float) -> float:
        """
        Update the conductance based on material-specific physics.
        
        Args:
            target_conductance: Desired conductance value
            voltage: Applied voltage
            dt: Time step
            
        Returns:
            Updated conductance value
        """
        raise NotImplementedError("Subclasses must implement update_conductance")


class HfO2RRAMModel(MaterialSpecificModel):
    """
    Physics-based model for HfO2-based RRAM (VCM mechanism).
    """
    
    def __init__(self, params: Optional[MaterialParameters] = None):
        default_params = MaterialParameters(
            low_resistance=50.0,
            high_resistance=50000.0,
            init_conductance=5e-5,
            filament_radius=2.2e-9,
            oxygen_vacancy_density=8e24,
            activation_energy=0.65,
            switching_mechanism=SwitchingMechanism.VCM,
            temperature_coefficient=0.0035
        )
        
        if params:
            # Override defaults with provided parameters
            for field in params.__dataclass_fields__:
                setattr(default_params, field, getattr(params, field))
        
        super().__init__(default_params)
    
    def update_conductance(self, target_conductance: float, voltage: float, dt: float) -> float:
        """Update conductance for HfO2 RRAM."""
        # Apply temperature effects
        temp_adjusted = self.get_conductance_with_temperature(target_conductance)
        
        # Apply aging effects (migration of oxygen vacancies)
        aging_drift = self.get_aging_effect()
        age_adjusted = temp_adjusted + aging_drift
        
        # Add process and cycle-to-cycle variations
        process_var = self.get_process_variation()
        ctc_var = self.get_cycle_to_cycle_variation()
        varied_conductance = age_adjusted * (1.0 + process_var) * (1.0 + ctc_var)
        
        # Simulate oxygen vacancy migration (simplified)
        k_boltzmann = 8.617e-5  # eV/K
        field_factor = abs(voltage) / 10.0  # Normalize voltage to field effect
        
        # Migration rate based on voltage and temperature
        migration_rate = self.params.attempt_freq * np.exp(
            -self.params.activation_energy / (k_boltzmann * self.temperature)
        )
        
        # Change in conductance based on voltage polarity and migration
        # In VCM, oxygen vacancies move to form/constrict conductive filaments
        if voltage > 0:  # Set operation (form filament)
            # More vacancies migrate at higher fields
            delta_g = 1e-6 * field_factor * migration_rate * dt
            new_conductance = varied_conductance + delta_g
        elif voltage < 0:  # Reset operation (constrict filament)
            # Vacancies migrate away, reducing conductance
            delta_g = 1e-6 * field_factor * migration_rate * dt
            new_conductance = max(varied_conductance - delta_g, 1.0 / self.params.high_resistance)
        else:  # No voltage, possible drift
            thermal_drift = np.random.normal(0, 1e-8) * dt
            new_conductance = varied_conductance + thermal_drift
        
        # Apply physical limits
        new_conductance = np.clip(
            new_conductance,
            1.0 / self.params.high_resistance,  # Min conductance
            1.0 / self.params.low_resistance    # Max conductance
        )
        
        self.current_conductance = new_conductance
        self.cycle_count += 1
        self.switching_history.append({
            'time': self.time,
            'voltage': voltage,
            'conductance': new_conductance,
            'cycle': self.cycle_count
        })
        
        return new_conductance


class TaOxRRAMModel(MaterialSpecificModel):
    """
    Physics-based model for TaOx-based RRAM (VCM mechanism).
    """
    
    def __init__(self, params: Optional[MaterialParameters] = None):
        default_params = MaterialParameters(
            low_resistance=100.0,
            high_resistance=100000.0,
            init_conductance=1e-4,
            filament_radius=2.5e-9,
            oxygen_vacancy_density=1.2e25,
            activation_energy=0.75,
            switching_mechanism=SwitchingMechanism.VCM,
            temperature_coefficient=0.0045,
            cycle_to_cycle=0.06
        )
        
        if params:
            # Override defaults with provided parameters
            for field in params.__dataclass_fields__:
                setattr(default_params, field, getattr(params, field))
        
        super().__init__(default_params)
    
    def update_conductance(self, target_conductance: float, voltage: float, dt: float) -> float:
        """Update conductance for TaOx RRAM."""
        # Apply all base effects
        temp_adjusted = self.get_conductance_with_temperature(target_conductance)
        aging_drift = self.get_aging_effect()
        age_adjusted = temp_adjusted + aging_drift
        
        process_var = self.get_process_variation()
        ctc_var = self.get_cycle_to_cycle_variation()
        varied_conductance = age_adjusted * (1.0 + process_var) * (1.0 + ctc_var)
        
        # TaOx specific physics
        k_boltzmann = 8.617e-5  # eV/K
        field_factor = abs(voltage) / 5.0  # Different normalization for TaOx
        
        # Migration rate with TaOx-specific activation energy
        migration_rate = self.params.attempt_freq * np.exp(
            -self.params.activation_energy / (k_boltzmann * self.temperature)
        )
        
        # TaOx has good linearity in analog switching
        if voltage > 0.5:  # Set operation
            # Linear conductance increase in TaOx
            delta_g = 1e-6 * field_factor * migration_rate * dt * np.sqrt(abs(voltage))
            new_conductance = varied_conductance + delta_g
        elif voltage < -0.5:  # Reset operation
            # Linear conductance decrease
            delta_g = 1e-6 * field_factor * migration_rate * dt * np.sqrt(abs(voltage))
            new_conductance = max(varied_conductance - delta_g, 1.0 / self.params.high_resistance)
        else:  # Sub-threshold, small drift
            thermal_drift = np.random.normal(0, 5e-9) * dt
            new_conductance = varied_conductance + thermal_drift
        
        # Apply limits
        new_conductance = np.clip(
            new_conductance,
            1.0 / self.params.high_resistance,
            1.0 / self.params.low_resistance
        )
        
        self.current_conductance = new_conductance
        self.cycle_count += 1
        self.switching_history.append({
            'time': self.time,
            'voltage': voltage,
            'conductance': new_conductance,
            'cycle': self.cycle_count
        })
        
        return new_conductance


class TiO2RRAMModel(MaterialSpecificModel):
    """
    Physics-based model for TiO2-based RRAM (VCM mechanism).
    """
    
    def __init__(self, params: Optional[MaterialParameters] = None):
        default_params = MaterialParameters(
            low_resistance=200.0,
            high_resistance=20000.0,
            init_conductance=2e-4,
            filament_radius=3.0e-9,
            oxygen_vacancy_density=6e24,
            activation_energy=0.8,
            switching_mechanism=SwitchingMechanism.VCM,
            temperature_coefficient=0.0025,
            cycle_to_cycle=0.03,
            retention_time=1e8  # Better retention for TiO2
        )
        
        if params:
            # Override defaults with provided parameters
            for field in params.__dataclass_fields__:
                setattr(default_params, field, getattr(params, field))
        
        super().__init__(default_params)
    
    def update_conductance(self, target_conductance: float, voltage: float, dt: float) -> float:
        """Update conductance for TiO2 RRAM."""
        # Apply base effects
        temp_adjusted = self.get_conductance_with_temperature(target_conductance)
        aging_drift = self.get_aging_effect()
        age_adjusted = temp_adjusted + aging_drift
        
        process_var = self.get_process_variation()
        ctc_var = self.get_cycle_to_cycle_variation()
        varied_conductance = age_adjusted * (1.0 + process_var) * (1.0 + ctc_var)
        
        # TiO2 specific physics (slower switching, better retention)
        k_boltzmann = 8.617e-5  # eV/K
        field_factor = abs(voltage) / 3.0  # Lower threshold for TiO2
        
        # Migration rate with higher activation energy
        migration_rate = self.params.attempt_freq * np.exp(
            -self.params.activation_energy / (k_boltzmann * self.temperature)
        )
        
        # TiO2 switching is more gradual
        if voltage > 0.3:  # Set operation
            delta_g = 5e-7 * field_factor * migration_rate * dt
            new_conductance = varied_conductance + delta_g
        elif voltage < -0.3:  # Reset operation
            delta_g = 5e-7 * field_factor * migration_rate * dt
            new_conductance = max(varied_conductance - delta_g, 1.0 / self.params.high_resistance)
        else:  # Stable state
            # Very small drift for better retention
            thermal_drift = np.random.normal(0, 1e-9) * dt
            new_conductance = varied_conductance + thermal_drift
        
        # Apply limits
        new_conductance = np.clip(
            new_conductance,
            1.0 / self.params.high_resistance,
            1.0 / self.params.low_resistance
        )
        
        self.current_conductance = new_conductance
        self.cycle_count += 1
        self.switching_history.append({
            'time': self.time,
            'voltage': voltage,
            'conductance': new_conductance,
            'cycle': self.cycle_count
        })
        
        return new_conductance


class NiOxRRAMModel(MaterialSpecificModel):
    """
    Physics-based model for NiOx-based RRAM (ECM mechanism).
    """
    
    def __init__(self, params: Optional[MaterialParameters] = None):
        default_params = MaterialParameters(
            low_resistance=80.0,
            high_resistance=80000.0,
            init_conductance=8e-5,
            filament_radius=1.8e-9,
            metal_ion_concentration=1.5e25,  # For ECM
            activation_energy=0.55,
            switching_mechanism=SwitchingMechanism.ECM,
            temperature_coefficient=0.005,
            cycle_to_cycle=0.07
        )
        
        if params:
            # Override defaults with provided parameters
            for field in params.__dataclass_fields__:
                setattr(default_params, field, getattr(params, field))
        
        super().__init__(default_params)
    
    def update_conductance(self, target_conductance: float, voltage: float, dt: float) -> float:
        """Update conductance for NiOx RRAM (ECM mechanism)."""
        # Apply base effects
        temp_adjusted = self.get_conductance_with_temperature(target_conductance)
        aging_drift = self.get_aging_effect()
        age_adjusted = temp_adjusted + aging_drift
        
        process_var = self.get_process_variation()
        ctc_var = self.get_cycle_to_cycle_variation()
        varied_conductance = age_adjusted * (1.0 + process_var) * (1.0 + ctc_var)
        
        # NiOx (ECM) specific physics - metal filament formation
        k_boltzmann = 8.617e-5  # eV/K
        field_factor = abs(voltage) / 1.0  # Lower voltage needed for metal migration
        
        # Migration rate with lower activation energy for metal ions
        migration_rate = self.params.attempt_freq * np.exp(
            -self.params.activation_energy / (k_boltzmann * self.temperature)
        )
        
        # In ECM, metal ions form/constrict the filament
        if voltage > 0:  # Form metal filament
            # Metal ion migration is typically faster
            delta_g = 2e-6 * field_factor * migration_rate * dt
            new_conductance = varied_conductance + delta_g
        elif voltage < 0:  # Dissolve metal filament
            delta_g = 2e-6 * field_factor * migration_rate * dt
            new_conductance = max(varied_conductance - delta_g, 1.0 / self.params.high_resistance)
        else:  # No voltage
            thermal_drift = np.random.normal(0, 2e-8) * dt
            new_conductance = varied_conductance + thermal_drift
        
        # Apply limits
        new_conductance = np.clip(
            new_conductance,
            1.0 / self.params.high_resistance,
            1.0 / self.params.low_resistance
        )
        
        self.current_conductance = new_conductance
        self.cycle_count += 1
        self.switching_history.append({
            'time': self.time,
            'voltage': voltage,
            'conductance': new_conductance,
            'cycle': self.cycle_count
        })
        
        return new_conductance


class CoOxRRAMModel(MaterialSpecificModel):
    """
    Physics-based model for CoOx-based RRAM (Mixed mechanism).
    """
    
    def __init__(self, params: Optional[MaterialParameters] = None):
        default_params = MaterialParameters(
            low_resistance=150.0,
            high_resistance=150000.0,
            init_conductance=1.5e-4,
            filament_radius=2.7e-9,
            oxygen_vacancy_density=9e24,
            metal_ion_concentration=8e23,
            activation_energy=0.72,
            switching_mechanism=SwitchingMechanism.MIXED,
            temperature_coefficient=0.004,
            cycle_to_cycle=0.05
        )
        
        if params:
            # Override defaults with provided parameters
            for field in params.__dataclass_fields__:
                setattr(default_params, field, getattr(params, field))
        
        super().__init__(default_params)
    
    def update_conductance(self, target_conductance: float, voltage: float, dt: float) -> float:
        """Update conductance for CoOx RRAM (Mixed mechanism)."""
        # Apply base effects
        temp_adjusted = self.get_conductance_with_temperature(target_conductance)
        aging_drift = self.get_aging_effect()
        age_adjusted = temp_adjusted + aging_drift
        
        process_var = self.get_process_variation()
        ctc_var = self.get_cycle_to_cycle_variation()
        varied_conductance = age_adjusted * (1.0 + process_var) * (1.0 + ctc_var)
        
        # CoOx (Mixed) specific physics - both vacancy and metal migration
        k_boltzmann = 8.617e-5  # eV/K
        field_factor = abs(voltage) / 2.0  # Mixed mechanism voltage
        
        # Combined migration rates
        vcm_rate = self.params.attempt_freq * np.exp(
            -self.params.activation_energy / (k_boltzmann * self.temperature)
        )
        ecm_rate = self.params.attempt_freq * np.exp(
            -0.6 / (k_boltzmann * self.temperature)  # Different activation for metal
        )
        
        # Mixed mechanism: both vacancy and metal contribute
        if voltage > 0:  # Set operation
            delta_g = (1e-6 * 0.7 * vcm_rate + 1e-6 * 0.3 * ecm_rate) * field_factor * dt
            new_conductance = varied_conductance + delta_g
        elif voltage < 0:  # Reset operation
            delta_g = (1e-6 * 0.7 * vcm_rate + 1e-6 * 0.3 * ecm_rate) * field_factor * dt
            new_conductance = max(varied_conductance - delta_g, 1.0 / self.params.high_resistance)
        else:  # Equilibrium
            thermal_drift = np.random.normal(0, 1.5e-8) * dt
            new_conductance = varied_conductance + thermal_drift
        
        # Apply limits
        new_conductance = np.clip(
            new_conductance,
            1.0 / self.params.high_resistance,
            1.0 / self.params.low_resistance
        )
        
        self.current_conductance = new_conductance
        self.cycle_count += 1
        self.switching_history.append({
            'time': self.time,
            'voltage': voltage,
            'conductance': new_conductance,
            'cycle': self.cycle_count
        })
        
        return new_conductance


class AdvancedMaterialsSimulator:
    """
    Advanced simulator that can handle multiple RRAM materials and their properties.
    """
    
    def __init__(self):
        self.material_models = {
            'HfO2': HfO2RRAMModel,
            'TaOx': TaOxRRAMModel,
            'TiO2': TiO2RRAMModel,
            'NiOx': NiOxRRAMModel,
            'CoOx': CoOxRRAMModel
        }
        self.simulation_history = []
    
    def create_model(self, 
                    material_type: str, 
                    params: Optional[MaterialParameters] = None) -> MaterialSpecificModel:
        """
        Create a material-specific model.
        
        Args:
            material_type: Type of material ('HfO2', 'TaOx', etc.)
            params: Optional custom parameters
            
        Returns:
            Initialized material-specific model
        """
        if material_type not in self.material_models:
            raise ValueError(f"Material type {material_type} not supported")
        
        model_class = self.material_models[material_type]
        return model_class(params)
    
    def simulate_multiple_cycles(self,
                               model: MaterialSpecificModel,
                               voltage_sequence: List[Tuple[float, float]],  # (voltage, duration)
                               initial_temperature: float = 300.0) -> List[Dict]:
        """
        Simulate multiple switching cycles for a model.
        
        Args:
            model: Material-specific model to simulate
            voltage_sequence: List of (voltage, duration) tuples
            initial_temperature: Starting temperature in Kelvin
            
        Returns:
            List of state changes during simulation
        """
        model.update_temperature(initial_temperature)
        history = []
        
        for voltage, duration in voltage_sequence:
            # Calculate number of small time steps
            dt = 1e-9  # Use 1ns steps
            steps = int(duration / dt)
            
            for step in range(steps):
                new_conductance = model.update_conductance(
                    model.current_conductance,
                    voltage,
                    dt
                )
                
                # Record state
                state = {
                    'time': model.time,
                    'voltage': voltage,
                    'conductance': new_conductance,
                    'cycle': model.cycle_count,
                    'temperature': model.temperature
                }
                history.append(state)
                
                # Update time for the model
                model.update_time(dt)
        
        # Add to overall history
        self.simulation_history.extend(history)
        
        return history
    
    def compare_materials(self,
                         materials: List[str],
                         voltage_profile: List[Tuple[float, float]],
                         temperature: float = 300.0) -> Dict[str, List[Dict]]:
        """
        Compare the behavior of different materials under same conditions.
        
        Args:
            materials: List of material types to compare
            voltage_profile: Voltage sequence to apply to all materials
            temperature: Temperature for simulation
            
        Returns:
            Dictionary mapping material names to their simulation histories
        """
        results = {}
        
        for material in materials:
            model = self.create_model(material)
            history = self.simulate_multiple_cycles(model, voltage_profile, temperature)
            results[material] = history
        
        return results
    
    def predict_device_lifetime(self,
                               model: MaterialSpecificModel,
                               use_conditions: Dict[str, float]) -> float:
        """
        Predict the lifetime of an RRAM device based on usage conditions.
        
        Args:
            model: Material-specific model
            use_conditions: Dictionary with usage conditions (temperature, voltage, cycles, etc.)
            
        Returns:
            Predicted lifetime in seconds
        """
        # Extract use conditions
        temp = use_conditions.get('temperature', 300.0)
        avg_voltage = use_conditions.get('avg_voltage', 1.0)
        cycles_per_day = use_conditions.get('cycles_per_day', 1000)
        
        # Calculate acceleration factor for temperature (Arrhenius equation)
        k_boltzmann = 8.617e-5  # eV/K
        acceleration_factor = np.exp(
            model.params.activation_energy * (1.0/300.0 - 1.0/temp) / k_boltzmann
        )
        
        # Calculate degradation rate adjusted for use conditions
        base_degradation_rate = model.params.degradation_rate
        voltage_effect = 1.0 + (abs(avg_voltage) - 1.0) * 0.1  # Higher voltage = faster degradation
        effective_degradation_rate = base_degradation_rate * acceleration_factor * voltage_effect
        
        # Calculate cycles until failure (define failure as 50% change from initial)
        initial_conductance = model.params.init_conductance
        failure_threshold = 0.5  # 50% change from initial value
        
        # Calculate time to reach failure threshold
        if effective_degradation_rate > 0:
            time_to_failure = failure_threshold / effective_degradation_rate
        else:
            time_to_failure = float('inf')
        
        # Account for cycling effects if applicable
        cycle_degradation_factor = use_conditions.get('cycle_degradation_factor', 1.0)
        adjusted_lifetime = time_to_failure / cycle_degradation_factor
        
        return adjusted_lifetime


def get_material_recommendation(problem_type: str,
                              constraints: Dict[str, float]) -> str:
    """
    Recommend the best RRAM material for a specific application.
    
    Args:
        problem_type: Type of problem ('high_speed', 'low_power', 'analog', etc.)
        constraints: Constraints like 'max_temperature', 'min_retention', etc.
        
    Returns:
        Recommended material type
    """
    if problem_type == 'high_speed':
        # Need fast switching - HfO2 or NiOx are good
        if constraints.get('max_temperature', 400) > 350:
            return 'NiOx'  # Metal filaments switch faster
        else:
            return 'HfO2'  # Good speed at moderate temp
    
    elif problem_type == 'low_power':
        # Need good retention and low drift - TiO2 is best
        return 'TiO2'
    
    elif problem_type == 'analog':
        # Need linear conductance change - TaOx is best
        return 'TaOx'
    
    elif problem_type == 'high_density':
        # Need small filaments - HfO2 has small filaments
        return 'HfO2'
    
    else:
        # Default to HfO2 for general applications
        return 'HfO2'


def demonstrate_material_simulation():
    """
    Demonstrate the advanced materials simulation capabilities.
    """
    print("Advanced Materials Simulation Demonstration")
    print("="*50)
    
    # Create simulator
    simulator = AdvancedMaterialsSimulator()
    
    # Define voltage sequence for testing
    voltage_profile = [
        (1.5, 1e-8),   # Set pulse
        (0.0, 1e-7),   # Hold
        (-1.2, 1e-8),  # Reset pulse
        (0.0, 1e-7),   # Hold
    ]
    
    # Compare different materials
    materials_to_compare = ['HfO2', 'TaOx', 'TiO2']
    results = simulator.compare_materials(materials_to_compare, voltage_profile)
    
    print(f"Comparing switching behavior of {materials_to_compare}:")
    for material, history in results.items():
        if history:
            final_conductance = history[-1]['conductance']
            print(f"  {material}: Final conductance = {final_conductance:.2e} S")
    
    # Create specific model and simulate
    hfo2_model = simulator.create_model('HfO2')
    
    # Simulate some switching cycles
    test_voltage_sequence = [
        (1.2, 1e-8),   # Set
        (0.0, 1e-6),   # Hold
        (-1.0, 1e-8),  # Reset
        (0.0, 1e-6),   # Hold
    ]
    
    history = simulator.simulate_multiple_cycles(hfo2_model, test_voltage_sequence)
    print(f"\nHfO2 model switching history: {len(history)} time steps recorded")
    
    # Predict lifetime under specific use conditions
    use_conditions = {
        'temperature': 350.0,      # Kelvin
        'avg_voltage': 0.8,        # V
        'cycles_per_day': 1000     # Number of switching cycles per day
    }
    
    lifetime = simulator.predict_device_lifetime(hfo2_model, use_conditions)
    print(f"Predicted HfO2 device lifetime: {lifetime/86400:.2f} days")
    
    # Get material recommendation
    recommendation = get_material_recommendation('analog', {})
    print(f"Recommended material for analog applications: {recommendation}")
    
    print("\nAdvanced materials simulation demonstration completed!")


if __name__ == "__main__":
    demonstrate_material_simulation()