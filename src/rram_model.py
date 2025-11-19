import numpy as np
from typing import List, Optional
import sys
import os
# Add the project root to path to import advanced models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import performance optimization tools
try:
    from .performance_optimization import PerformanceOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Default 8 conductance levels (3-bit) from the paper, in Siemens
DEFAULT_CONDUCTANCE_LEVELS = np.array([0.5, 5, 10, 15, 20, 25, 30, 35]) * 1e-6

def create_rram_matrix(n: int,
                       conductance_levels: List[float] = DEFAULT_CONDUCTANCE_LEVELS,
                       variability: float = 0.05,
                       stuck_fault_prob: float = 0.01,
                       line_resistance: float = 1.7e-3,
                       temperature: float = 300.0,
                       time_since_programming: float = 0.0,
                       temp_coeff: float = 0.002,
                       activation_energy: float = 0.5,
                       use_advanced_physics: bool = False,
                       material: str = 'HfO2',
                       device_area: float = 1e-12,
                       use_material_specific: bool = False,
                       ecm_vcm_ratio: float = 0.5) -> np.ndarray:
    """
    Create an n x n RRAM conductance matrix with discrete levels, variability, 
    stuck faults, line resistance effects, and temperature/time-dependent drift.

    Args:
        n: Matrix size
        conductance_levels: A list of discrete conductance levels (S)
        variability: Relative standard deviation for conductance deviations
        stuck_fault_prob: Probability of stuck-at fault per cell
        line_resistance: Wire resistance per adjacent device (Ω)
        temperature: Current temperature in Kelvin (default: 300K)
        time_since_programming: Time elapsed since programming in seconds (default: 0)
        temp_coeff: Temperature coefficient for instantaneous compensation (default: 0.002)
        activation_energy: Activation energy for time-dependent drift (eV) (default: 0.5)
        use_advanced_physics: Whether to use advanced physics-based models (default: False)
        material: RRAM material type ('HfO2', 'TaOx', 'TiO2', etc.) (default: 'HfO2')
        device_area: Physical area of each RRAM device (m²) (default: 1e-12)
        use_material_specific: Whether to use material-specific physics models (default: False)
        ecm_vcm_ratio: Ratio of ECM to VCM for mixed-mechanism materials (0.0-1.0)

    Returns:
        Conductance matrix G
    """
    if use_material_specific:
        # Use the new material-specific physics models
        G = create_material_specific_rram_matrix(
            n=n,
            material=material,
            device_area=device_area,
            temperature=temperature,
            stuck_fault_prob=stuck_fault_prob,
            time_since_programming=time_since_programming,
            ecm_vcm_ratio=ecm_vcm_ratio
        )
    elif use_advanced_physics:
        # Use the older advanced physics-based model (general)
        try:
            from .advanced_rram_models import AdvancedRRAMModel, RRAMMaterial
            
            # Map string material to enum
            material_map = {
                'HfO2': RRAMMaterial.HFO2,
                'TaOx': RRAMMaterial.TAOX,
                'TiO2': RRAMMaterial.TIO2,
                'NiOx': RRAMMaterial.NIOX,
                'CoOx': RRAMMaterial.COOX
            }
            material_enum = material_map.get(material, RRAMMaterial.HFO2)
            
            # Create individual devices using advanced model
            G = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    device = AdvancedRRAMModel(
                        material=material_enum,
                        device_area=device_area,
                        temperature=temperature,
                        ecm_vcm_ratio=ecm_vcm_ratio
                    )
                    # Apply some programming voltage to set a state
                    # For simplicity, apply a voltage that favors a particular state
                    voltage_magnitude = {
                        'HfO2': (-1.5, 2.0), 'TaOx': (-1.2, 2.0), 'TiO2': (-1.8, 2.2),
                        'NiOx': (-0.8, 1.5), 'CoOx': (-1.0, 2.0)
                    }.get(material, (-1.5, 2.0))
                    
                    if np.random.random() > 0.5:
                        device.apply_voltage(voltage_magnitude[0], 1e-8)  # SET attempt
                    else:
                        device.apply_voltage(voltage_magnitude[1], 1e-8)   # RESET attempt
                    G[i, j] = device.get_conductance()
        except ImportError:
            # Fallback to basic model if advanced models not available
            print("Warning: Advanced RRAM models not available, using basic model")
            G = _create_basic_rram_matrix(n, conductance_levels, variability, 
                                         stuck_fault_prob, line_resistance, 
                                         temperature, time_since_programming, 
                                         temp_coeff, activation_energy)
    else:
        # Use the original basic model
        G = _create_basic_rram_matrix(n, conductance_levels, variability, 
                                     stuck_fault_prob, line_resistance, 
                                     temperature, time_since_programming, 
                                     temp_coeff, activation_energy)

    return G


def _create_basic_rram_matrix(n: int,
                              conductance_levels: List[float],
                              variability: float,
                              stuck_fault_prob: float,
                              line_resistance: float,
                              temperature: float,
                              time_since_programming: float,
                              temp_coeff: float,
                              activation_energy: float) -> np.ndarray:
    """Internal function to create basic RRAM matrix (original implementation)."""
    # Base matrix with discrete conductance levels
    level_indices = np.random.randint(0, len(conductance_levels), size=(n, n))
    G = np.array(conductance_levels)[level_indices]

    # Add conductance variability (±variability)
    noise = np.random.normal(1.0, variability, (n, n))
    G *= noise

    # Add stuck-at faults (stuck at 0 for simplicity)
    stuck_mask = np.random.random((n, n)) < stuck_fault_prob
    G[stuck_mask] = 0.0

    # Approximate line resistance effect (simplified: add resistance in series)
    # For crossbar, line resistance affects the effective conductance
    # Simplified model: reduce conductance by factor related to line resistance
    if line_resistance > 0:
        # Rough approximation: effective G_eff = G / (1 + G * R_line * n) or something
        # For simplicity, add a small perturbation
        line_effect = np.random.normal(1.0, 0.01, (n, n))  # Small variation
        G *= line_effect

    # Apply temperature effects
    from .temperature_compensation import temperature_coefficient_model, arrhenius_drift_model
    
    # Apply instantaneous temperature coefficient model
    if temperature != 300.0:  # Only apply if not at reference temperature
        G = temperature_coefficient_model(G, temp_op=temperature, temp_coeff=temp_coeff)

    # Apply time-dependent drift
    if time_since_programming > 0:
        G = arrhenius_drift_model(G, 0.0, time_since_programming, temperature, activation_energy)

    return G


def create_material_specific_rram_matrix(n: int,
                                       material: str = 'HfO2',
                                       device_area: float = 1e-12,
                                       temperature: float = 300.0,
                                       stuck_fault_prob: float = 0.01,
                                       time_since_programming: float = 0.0,
                                       ecm_vcm_ratio: float = 0.5) -> np.ndarray:
    """
    Create an n x n RRAM conductance matrix using material-specific models.
    
    This function uses the advanced material-specific physics models for different
    RRAM materials (HfO2, TaOx, TiO2, NiOx, CoOx) with their specific switching
    mechanisms and characteristics.
    
    Args:
        n: Matrix size
        material: RRAM material type ('HfO2', 'TaOx', 'TiO2', 'NiOx', 'CoOx')
        device_area: Physical area of each RRAM device (m²)
        temperature: Operating temperature in Kelvin
        stuck_fault_prob: Probability of stuck-at fault per cell
        time_since_programming: Time elapsed since programming in seconds
        ecm_vcm_ratio: Ratio of ECM to VCM for mixed-mechanism materials
        
    Returns:
        Conductance matrix G with material-specific characteristics
    """
    from .advanced_rram_models import AdvancedRRAMModel, RRAMMaterial
    
    # Map string material to enum
    material_map = {
        'HfO2': RRAMMaterial.HFO2,
        'TaOx': RRAMMaterial.TAOX,
        'TiO2': RRAMMaterial.TIO2,
        'NiOx': RRAMMaterial.NIOX,
        'CoOx': RRAMMaterial.COOX
    }
    material_enum = material_map.get(material, RRAMMaterial.HFO2)
    
    # Create individual devices using advanced material-specific model
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            device = AdvancedRRAMModel(
                material=material_enum,
                device_area=device_area,
                temperature=temperature,
                ecm_vcm_ratio=ecm_vcm_ratio
            )
            
            # Apply some conditioning voltage to set a base state
            # This simulates the programming state of the device
            if np.random.random() > 0.5:
                device.apply_voltage(-1.5 if material in ['HfO2', 'TiO2', 'CoOx'] 
                                   else -0.8 if material == 'NiOx' 
                                   else -1.2, 1e-8)  # SET attempt
            else:
                device.apply_voltage(2.0 if material in ['HfO2', 'TiO2', 'CoOx'] 
                                   else 1.5 if material == 'NiOx' 
                                   else 2.0, 1e-8)   # RESET attempt
            
            G[i, j] = device.get_conductance()
    
    # Add stuck-at faults (stuck at 0 for simplicity)
    stuck_mask = np.random.random((n, n)) < stuck_fault_prob
    G[stuck_mask] = 0.0
    
    # Apply time-dependent effects if applicable
    if time_since_programming > 0:
        from .temperature_compensation import arrhenius_drift_model
        G = arrhenius_drift_model(G, 0.0, time_since_programming, temperature, 0.5)
    
    return G

def mvm(G: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Perform matrix-vector multiplication with conductance matrix.

    Args:
        G: Conductance matrix
        x: Input vector

    Returns:
        Output vector y = G @ x
    """
    return G @ x