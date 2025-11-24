"""
Real-time Adaptive Systems for RRAM.

This module provides systems that can dynamically adjust their parameters
during operation based on changing conditions, temperature, or aging of
the RRAM devices. These systems continuously monitor and adapt to maintain
optimal performance despite environmental changes and device degradation.
"""
import numpy as np
import time
import threading
import queue
import warnings
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass
from collections import deque
from enum import Enum
import json
from datetime import datetime
import statistics

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV
    from .neuromorphic_modules import RRAMSpikingLayer, RRAMReservoirComputer
    from .optimization_algorithms import RRAMAwareOptimizer
    from .benchmarking_suite import RRAMBenchmarkSuite
    from .advanced_materials_simulation import AdvancedMaterialsSimulator
    from .edge_computing_integration import EdgeRRAMOptimizer
    from .performance_profiling import PerformanceProfiler
    from .gpu_simulation_acceleration import GPUSimulationAccelerator
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class AdaptiveSystemMode(Enum):
    """Modes of operation for adaptive systems."""
    MONITOR_ONLY = "monitor_only"
    AUTO_ADJUST = "auto_adjust"
    LEARN_AND_ADAPT = "learn_and_adapt"
    PREDICTIVE = "predictive"


class AdaptiveParameter(Enum):
    """Parameters that can be adapted."""
    PRECISION_BITS = "precision_bits"
    RELAXATION_FACTOR = "relaxation_factor"
    NOISE_LEVEL = "noise_level"
    ITERATION_LIMIT = "iteration_limit"
    TEMPERATURE_COMPENSATION = "temperature_compensation"
    VOLTAGE_BIAS = "voltage_bias"
    TIMESTEP = "timestep"


@dataclass
class AdaptiveState:
    """Represents the current state of adaptive system."""
    timestamp: float
    current_values: Dict[AdaptiveParameter, float]
    target_values: Dict[AdaptiveParameter, float]
    error_metrics: Dict[str, float]
    environmental_conditions: Dict[str, float]
    performance_metrics: Dict[str, float]
    adaptation_history: List[Dict[str, Any]]


class EnvironmentSensor:
    """
    Simulated environmental sensor for monitoring temperature,
    aging, and other environmental factors.
    """
    
    def __init__(self):
        self.temperature = 300.0  # Kelvin
        self.voltage = 3.3  # Volts
        self.humidity = 0.5  # 50% humidity
        self.last_update = time.time()
    
    def read_temperature(self) -> float:
        """Read current temperature with some fluctuation."""
        # Simulate temperature fluctuations
        fluctuation = np.random.normal(0, 0.5)  # ±0.5K fluctuation
        self.temperature += fluctuation
        return self.temperature
    
    def read_voltage(self) -> float:
        """Read current voltage with some fluctuation."""
        # Simulate voltage fluctuations
        fluctuation = np.random.normal(0, 0.01)  # ±10mV fluctuation
        self.voltage += fluctuation
        return self.voltage
    
    def read_humidity(self) -> float:
        """Read current humidity."""
        # Slow drift in humidity
        drift = np.random.normal(0, 0.001)
        self.humidity = np.clip(self.humidity + drift, 0.1, 0.9)  # 10% to 90%
        return self.humidity
    
    def update_environment(self) -> Dict[str, float]:
        """Update environment and return current conditions."""
        self.temperature = self.read_temperature()
        self.voltage = self.read_voltage()
        self.humidity = self.read_humidity()
        self.last_update = time.time()
        
        return {
            'temperature_k': self.temperature,
            'voltage_v': self.voltage,
            'humidity': self.humidity,
            'last_update': self.last_update
        }


class ParameterAdaptationController:
    """
    Controller for adapting RRAM parameters based on environmental
    conditions and performance metrics. Uses bounded history to prevent
    memory leaks during long-term operation.
    """

    def __init__(self,
                 initial_settings: Dict[AdaptiveParameter, float],
                 adaptation_rates: Optional[Dict[AdaptiveParameter, float]] = None):
        """
        Initialize the parameter adaptation controller.

        Args:
            initial_settings: Initial parameter values
            adaptation_rates: Rates of adaptation for each parameter
        """
        self.current_settings = initial_settings.copy()
        self.adaptation_rates = adaptation_rates or {
            AdaptiveParameter.PRECISION_BITS: 0.1,      # Small adjustments for precision
            AdaptiveParameter.RELAXATION_FACTOR: 0.05,  # Conservative for stability
            AdaptiveParameter.NOISE_LEVEL: 0.01,        # Small noise adjustments
            AdaptiveParameter.ITERATION_LIMIT: 1.0,     # Integer adjustments
            AdaptiveParameter.TEMPERATURE_COMPENSATION: 0.1,
            AdaptiveParameter.VOLTAGE_BIAS: 0.005,
            AdaptiveParameter.TIMESTEP: 0.001
        }
        self.adaptation_history = deque(maxlen=500)
        self.lock = threading.Lock()
    
    def update_parameter(self, 
                        param: AdaptiveParameter, 
                        new_value: float, 
                        reason: str = "") -> bool:
        """
        Update a parameter value.
        
        Args:
            param: Parameter to update
            new_value: New value for the parameter
            reason: Reason for the update
            
        Returns:
            True if update was successful, False otherwise
        """
        with self.lock:
            try:
                # Apply constraints based on parameter type
                constrained_value = self._apply_parameter_constraints(param, new_value)
                self.current_settings[param] = constrained_value

                # Log the update (automatically bounded by deque maxlen)
                self.adaptation_history.append({
                    'timestamp': time.time(),
                    'parameter': param.value,
                    'old_value': self.current_settings[param],
                    'new_value': constrained_value,
                    'reason': reason,
                    'delta': constrained_value - self.current_settings[param]
                })

                return True
            except Exception as e:
                print(f"Error updating parameter {param}: {e}")
                return False
    
    def _apply_parameter_constraints(self, 
                                   param: AdaptiveParameter, 
                                   value: float) -> float:
        """
        Apply constraints to parameter values.
        
        Args:
            param: Parameter to constrain
            value: Value to constrain
            
        Returns:
            Constrained value
        """
        if param == AdaptiveParameter.PRECISION_BITS:
            return np.clip(value, 2, 8)  # 2-8 bits precision
        elif param == AdaptiveParameter.RELAXATION_FACTOR:
            return np.clip(value, 0.1, 2.0)  # 0.1 to 2.0 for relaxation
        elif param == AdaptiveParameter.NOISE_LEVEL:
            return np.clip(value, 1e-6, 0.1)  # Noise between 1µ and 10%
        elif param == AdaptiveParameter.ITERATION_LIMIT:
            return np.clip(value, 1, 100)  # 1-100 iterations
        elif param == AdaptiveParameter.TEMPERATURE_COMPENSATION:
            return np.clip(value, 0.0, 1.0)  # 0-1 for compensation factor
        elif param == AdaptiveParameter.VOLTAGE_BIAS:
            return np.clip(value, -0.5, 0.5)  # ±0.5V bias
        elif param == AdaptiveParameter.TIMESTEP:
            return np.clip(value, 1e-9, 1e-3)  # 1ns to 1ms timestep
        else:
            return value
    
    def get_adaptation_suggestions(self, 
                                  error_metrics: Dict[str, float],
                                  env_conditions: Dict[str, float]) -> Dict[AdaptiveParameter, float]:
        """
        Get suggestions for parameter adaptations based on error and environment.
        
        Args:
            error_metrics: Current error metrics
            env_conditions: Current environmental conditions
            
        Returns:
            Dictionary of suggested parameter adjustments
        """
        suggestions = {}
        
        # Adjust precision based on accuracy errors
        if 'residual_norm' in error_metrics:
            residual = error_metrics['residual_norm']
            if residual > 1e-3:  # Accuracy issue
                current_precision = self.current_settings.get(AdaptiveParameter.PRECISION_BITS, 4)
                suggestions[AdaptiveParameter.PRECISION_BITS] = current_precision + 0.5
            elif residual < 1e-6:  # Possibly over-precise
                current_precision = self.current_settings.get(AdaptiveParameter.PRECISION_BITS, 4)
                suggestions[AdaptiveParameter.PRECISION_BITS] = current_precision - 0.2
        
        # Adjust relaxation factor based on convergence
        if 'convergence_rate' in error_metrics:
            conv_rate = error_metrics['convergence_rate']
            if conv_rate < 0.1:  # Slow convergence
                current_relax = self.current_settings.get(AdaptiveParameter.RELAXATION_FACTOR, 0.9)
                suggestions[AdaptiveParameter.RELAXATION_FACTOR] = min(1.5, current_relax * 1.1)
            elif conv_rate > 0.9:  # Fast convergence but possibly unstable
                current_relax = self.current_settings.get(AdaptiveParameter.RELAXATION_FACTOR, 0.9)
                suggestions[AdaptiveParameter.RELAXATION_FACTOR] = max(0.5, current_relax * 0.95)
        
        # Adjust for temperature
        if 'temperature_k' in env_conditions:
            temp_k = env_conditions['temperature_k']
            if temp_k > 350:  # High temperature - increase noise tolerance
                current_noise = self.current_settings.get(AdaptiveParameter.NOISE_LEVEL, 0.01)
                suggestions[AdaptiveParameter.NOISE_LEVEL] = min(0.05, current_noise + 0.002)
                # Decrease precision slightly for stability
                current_precision = self.current_settings.get(AdaptiveParameter.PRECISION_BITS, 4)
                suggestions[AdaptiveParameter.PRECISION_BITS] = max(2, current_precision - 0.1)
            elif temp_k < 280:  # Low temperature - can be more precise
                current_precision = self.current_settings.get(AdaptiveParameter.PRECISION_BITS, 4)
                suggestions[AdaptiveParameter.PRECISION_BITS] = min(8, current_precision + 0.2)
        
        # Adjust for aging (simulated by execution count)
        if 'execution_count' in error_metrics:
            exec_count = error_metrics['execution_count']
            if exec_count > 1000:  # Potential aging effects
                current_precision = self.current_settings.get(AdaptiveParameter.PRECISION_BITS, 4)
                suggestions[AdaptiveParameter.PRECISION_BITS] = max(2, current_precision - 0.1)
                # Increase noise tolerance for aged devices
                current_noise = self.current_settings.get(AdaptiveParameter.NOISE_LEVEL, 0.01)
                suggestions[AdaptiveParameter.NOISE_LEVEL] = min(0.05, current_noise + 0.001)
        
        return suggestions
    
    def get_current_settings(self) -> Dict[AdaptiveParameter, float]:
        """Get current parameter settings."""
        return self.current_settings.copy()


class RealTimeAdaptiveSystem:
    """
    Real-time adaptive system that monitors and adjusts RRAM parameters.
    Uses bounded history buffers to prevent memory leaks during long-term operation.
    """

    def __init__(self,
                 rram_interface: Optional[object] = None,
                 mode: AdaptiveSystemMode = AdaptiveSystemMode.AUTO_ADJUST,
                 adaptation_frequency: float = 1.0):
        """
        Initialize the real-time adaptive system.

        Args:
            rram_interface: Optional RRAM interface for hardware feedback
            mode: Operating mode for the adaptive system
            adaptation_frequency: How often to adapt parameters (Hz)
        """
        self.rram_interface = rram_interface
        self.mode = mode
        self.adaptation_frequency = adaptation_frequency
        self.environment_sensor = EnvironmentSensor()

        # Initialize parameter controller
        initial_settings = {
            AdaptiveParameter.PRECISION_BITS: 4.0,
            AdaptiveParameter.RELAXATION_FACTOR: 0.9,
            AdaptiveParameter.NOISE_LEVEL: 0.01,
            AdaptiveParameter.ITERATION_LIMIT: 15.0,
            AdaptiveParameter.TEMPERATURE_COMPENSATION: 0.1,
            AdaptiveParameter.VOLTAGE_BIAS: 0.0,
            AdaptiveParameter.TIMESTEP: 1e-6
        }

        self.param_controller = ParameterAdaptationController(initial_settings)

        self.performance_history = deque(maxlen=100)
        self.system_running = False
        self.adaptation_thread = None

        # Performance metrics
        self.metrics = {
            'residual_norm': 1.0,
            'convergence_rate': 0.5,
            'execution_time': 0.01,
            'accuracy': 0.99,
            'stability': 0.95,
            'execution_count': 0
        }
    
    def start_adaptation_loop(self):
        """Start the background adaptation loop."""
        if self.adaptation_thread and self.adaptation_thread.is_alive():
            return
        
        self.system_running = True
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop)
        self.adaptation_thread.daemon = True
        self.adaptation_thread.start()
        
        print(f"Started real-time adaptation loop (mode: {self.mode.value}, freq: {self.adaptation_frequency}Hz)")
    
    def stop_adaptation_loop(self):
        """Stop the background adaptation loop."""
        self.system_running = False
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=1.0)  # Wait up to 1 second
        print("Stopped real-time adaptation loop")
    
    def _adaptation_loop(self):
        """Background loop for continuous adaptation."""
        while self.system_running:
            try:
                # Monitor current state
                env_conditions = self.environment_sensor.update_environment()
                
                # Get performance metrics
                current_metrics = self.metrics.copy()
                
                if self.mode in [AdaptiveSystemMode.AUTO_ADJUST, AdaptiveSystemMode.LEARN_AND_ADAPT]:
                    # Get adaptation suggestions
                    suggestions = self.param_controller.get_adaptation_suggestions(
                        current_metrics, env_conditions
                    )
                    
                    # Apply adaptations based on the system mode
                    if self.mode == AdaptiveSystemMode.AUTO_ADJUST:
                        self._apply_adaptations(suggestions)
                    elif self.mode == AdaptiveSystemMode.LEARN_AND_ADAPT:
                        self._learn_from_performance_and_adapt(suggestions, env_conditions)
                
                # Sleep until next adaptation cycle
                time.sleep(1.0 / self.adaptation_frequency)
                
            except Exception as e:
                print(f"Error in adaptation loop: {e}")
                time.sleep(0.1)  # Brief pause before continuing
    
    def _apply_adaptations(self, suggestions: Dict[AdaptiveParameter, float]):
        """Apply suggested parameter adaptations."""
        for param, suggested_value in suggestions.items():
            current_value = self.param_controller.get_current_settings().get(param, 0)
            new_value = current_value + self.param_controller.adaptation_rates[param] * (suggested_value - current_value)
            
            success = self.param_controller.update_parameter(
                param, 
                new_value, 
                reason="Adaptation suggestion"
            )
            
            if success:
                print(f"Applied adaptation: {param.value} from {current_value:.3f} to {new_value:.3f}")
    
    def _learn_from_performance_and_adapt(self, 
                                         suggestions: Dict[AdaptiveParameter, float],
                                         env_conditions: Dict[str, float]):
        """Learn from performance and apply adaptive changes."""
        # This is where machine learning would come in
        # For now, we'll add a simple learning component that remembers 
        # which adaptations led to performance improvements
        
        # Store current state
        current_settings = self.param_controller.get_current_settings()
        current_performance = self.metrics.copy()
        current_env = env_conditions.copy()
        
        # Apply suggested adaptations (similar to regular adaptation but with learning)
        self._apply_adaptations(suggestions)

        # Update learning history (automatically bounded by deque maxlen)
        self.performance_history.append({
            'timestamp': time.time(),
            'settings_before': current_settings,
            'env_conditions': current_env,
            'suggestions_applied': list(suggestions.keys()),
            'performance_before': current_performance,
            'performance_after': self.metrics.copy()
        })
    
    def update_performance_metrics(self, 
                                  residual_norm: Optional[float] = None,
                                  execution_time: Optional[float] = None,
                                  accuracy: Optional[float] = None,
                                  **kwargs):
        """
        Update the system's performance metrics.
        
        Args:
            residual_norm: Current residual norm
            execution_time: Execution time for last operation
            accuracy: Accuracy of the computation
            **kwargs: Additional metrics
        """
        if residual_norm is not None:
            self.metrics['residual_norm'] = residual_norm
        
        if execution_time is not None:
            self.metrics['execution_time'] = execution_time
            
            # Calculate convergence rate (inverse of time, with upper bound)
            self.metrics['convergence_rate'] = min(1.0, 0.1 / (execution_time + 1e-6))
        
        if accuracy is not None:
            self.metrics['accuracy'] = accuracy
            # Stability calculated from accuracy and other factors
            stability = min(1.0, accuracy + 0.1)  # Simple stability model
            self.metrics['stability'] = stability
        
        # Update execution count
        self.metrics['execution_count'] = self.metrics.get('execution_count', 0) + 1
        
        # Incorporate any additional metrics
        for key, value in kwargs.items():
            self.metrics[key] = value
    
    def get_optimal_parameters(self, 
                              problem_characteristics: Dict[str, float]) -> Dict[AdaptiveParameter, float]:
        """
        Get optimal parameters for a specific problem.
        
        Args:
            problem_characteristics: Characteristics of the problem to solve
            
        Returns:
            Optimal parameter values
        """
        current_settings = self.param_controller.get_current_settings()
        env_conditions = self.environment_sensor.update_environment()
        
        # Adjust parameters based on problem characteristics and environment
        optimal_params = current_settings.copy()
        
        # Adjust for problem condition number
        if 'condition_number' in problem_characteristics:
            cond_num = problem_characteristics['condition_number']
            if cond_num > 1e6:  # Ill-conditioned
                optimal_params[AdaptiveParameter.RELAXATION_FACTOR] = 0.7
                optimal_params[AdaptiveParameter.PRECISION_BITS] = min(8, current_settings[AdaptiveParameter.PRECISION_BITS] + 1)
                optimal_params[AdaptiveParameter.ITERATION_LIMIT] = min(50, current_settings[AdaptiveParameter.ITERATION_LIMIT] + 10)
            elif cond_num < 1e3:  # Well-conditioned
                optimal_params[AdaptiveParameter.RELAXATION_FACTOR] = 1.1
                optimal_params[AdaptiveParameter.PRECISION_BITS] = max(2, current_settings[AdaptiveParameter.PRECISION_BITS] - 1)
        
        # Adjust for problem size
        if 'problem_size' in problem_characteristics:
            size = problem_characteristics['problem_size']
            if size > 100:  # Large problem
                # More aggressive parameters for large problems
                optimal_params[AdaptiveParameter.ITERATION_LIMIT] = min(100, current_settings[AdaptiveParameter.ITERATION_LIMIT] + 5)
            else:  # Small problem
                # More conservative for small problems
                optimal_params[AdaptiveParameter.ITERATION_LIMIT] = max(5, current_settings[AdaptiveParameter.ITERATION_LIMIT] - 2)
        
        # Adjust for temperature
        temp_k = env_conditions.get('temperature_k', 300.0)
        if temp_k > 350:
            # High temperature - be more conservative
            optimal_params[AdaptiveParameter.PRECISION_BITS] = max(2, current_settings[AdaptiveParameter.PRECISION_BITS] - 0.5)
            optimal_params[AdaptiveParameter.RELAXATION_FACTOR] = min(1.0, current_settings[AdaptiveParameter.RELAXATION_FACTOR] * 0.9)
        elif temp_k < 280:
            # Low temperature - can be more aggressive
            optimal_params[AdaptiveParameter.PRECISION_BITS] = min(8, current_settings[AdaptiveParameter.PRECISION_BITS] + 0.5)
        
        return optimal_params
    
    def adapt_for_problem(self, 
                         G: np.ndarray, 
                         b: np.ndarray,
                         custom_parameters: Optional[Dict[AdaptiveParameter, float]] = None) -> Dict[AdaptiveParameter, float]:
        """
        Adapt parameters specifically for solving Gx = b.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            custom_parameters: Custom parameter overrides
            
        Returns:
            Adapted parameters
        """
        # Analyze problem characteristics
        problem_char = {
            'condition_number': float(np.linalg.cond(G)) if np.linalg.cond(G) < 1e15 else 1e15,
            'problem_size': G.shape[0],
            'sparsity': float(np.count_nonzero(G) / G.size),
            'right_hand_side_norm': float(np.linalg.norm(b))
        }
        
        # Get optimal parameters for this problem
        optimal_params = self.get_optimal_parameters(problem_char)
        
        if custom_parameters:
            optimal_params.update(custom_parameters)
        
        # Update the controller with these parameters
        for param, value in optimal_params.items():
            self.param_controller.update_parameter(
                param, 
                value, 
                reason="Problem-specific adaptation"
            )
        
        return optimal_params


class AdaptiveHPINV:
    """
    Adaptive HP-INV that adjusts its parameters based on system state.
    """
    
    def __init__(self, 
                 adaptive_system: RealTimeAdaptiveSystem,
                 fallback_solver: Callable = hp_inv):
        """
        Initialize the adaptive HP-INV solver.
        
        Args:
            adaptive_system: Real-time adaptive system
            fallback_solver: Fallback solver if adaptation fails
        """
        self.adaptive_system = adaptive_system
        self.fallback_solver = fallback_solver
    
    def solve(self, 
              G: np.ndarray, 
              b: np.ndarray, 
              **kwargs) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Solve Gx = b using adaptive parameters.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            Solution from adaptive solver
        """
        start_time = time.time()
        
        # Adapt parameters for this specific problem
        adapted_params = self.adaptive_system.adapt_for_problem(G, b)
        
        # Convert AdaptiveParameter enum to HP-INV parameter names
        hp_inv_params = {
            'bits': int(adapted_params[AdaptiveParameter.PRECISION_BITS]),
            'relaxation_factor': adapted_params[AdaptiveParameter.RELAXATION_FACTOR],
            'lp_noise_std': adapted_params[AdaptiveParameter.NOISE_LEVEL],
            'max_iter': int(adapted_params[AdaptiveParameter.ITERATION_LIMIT]),
            'tol': 1e-6
        }
        
        # Override with any explicit parameters
        hp_inv_params.update(kwargs)
        
        try:
            solution, iterations, info = hp_inv(G, b, **hp_inv_params)
        except Exception:
            # Fallback to original solver if adaptation caused issues
            solution, iterations, info = self.fallback_solver(G, b, **kwargs)
        
        execution_time = time.time() - start_time
        
        # Update performance metrics
        residual = np.linalg.norm(G @ solution - b)
        self.adaptive_system.update_performance_metrics(
            residual_norm=residual,
            execution_time=execution_time,
            accuracy=1.0 / (1.0 + residual)  # Simple accuracy proxy
        )
        
        return solution, iterations, info


class AdaptiveNeuralNetwork:
    """
    Adaptive neural network using RRAM properties.
    """
    
    def __init__(self, 
                 adaptive_system: RealTimeAdaptiveSystem,
                 architecture: List[int],
                 rram_interface: Optional[object] = None):
        """
        Initialize adaptive neural network.
        
        Args:
            adaptive_system: Real-time adaptive system
            architecture: List of layer sizes [input, hidden1, ..., output]
            rram_interface: Optional RRAM interface
        """
        self.adaptive_system = adaptive_system
        self.architecture = architecture
        self.rram_interface = rram_interface
        self.layers = []
        self.build_network()
    
    def build_network(self):
        """Build the neural network layers."""
        for i in range(len(self.architecture) - 1):
            input_size = self.architecture[i]
            output_size = self.architecture[i+1]
            
            # Create spiking layer with RRAM properties
            layer = RRAMSpikingLayer(
                n_inputs=input_size,
                n_neurons=output_size,
                rram_interface=self.rram_interface
            )
            self.layers.append(layer)
    
    def forward(self, 
                input_signal: np.ndarray, 
                adapt_for_input: bool = True) -> np.ndarray:
        """
        Forward pass with adaptive behavior.
        
        Args:
            input_signal: Input signal
            adapt_for_input: Whether to adapt for this input
            
        Returns:
            Output signal
        """
        if adapt_for_input:
            # Adapt parameters based on input characteristics
            input_stats = {
                'mean': float(np.mean(input_signal)),
                'std': float(np.std(input_signal)),
                'magnitude': float(np.linalg.norm(input_signal))
            }
            
            # Simplified adaptation for neural network
            if input_stats['magnitude'] > 1.0:
                # Strong input - adjust for higher sensitivity
                self.adaptive_system.update_performance_metrics(
                    signal_magnitude=input_stats['magnitude']
                )
        
        # Process through layers
        output = input_signal
        for layer in self.layers:
            output = layer.forward(output)
        
        return output


def create_real_time_adaptive_system(
    rram_interface: Optional[object] = None,
    mode: AdaptiveSystemMode = AdaptiveSystemMode.AUTO_ADJUST
) -> Tuple[RealTimeAdaptiveSystem, AdaptiveHPINV]:
    """
    Create a real-time adaptive system with its components.
    
    Args:
        rram_interface: Optional RRAM interface
        mode: Operating mode for the adaptive system
        
    Returns:
        Tuple of (adaptive system, adaptive solver)
    """
    adaptive_system = RealTimeAdaptiveSystem(
        rram_interface=rram_interface,
        mode=mode
    )
    
    adaptive_solver = AdaptiveHPINV(adaptive_system)
    
    return adaptive_system, adaptive_solver


def demonstrate_adaptive_systems():
    """
    Demonstrate real-time adaptive systems capabilities.
    """
    print("Real-time Adaptive Systems Demonstration")
    print("="*50)
    
    # Create adaptive system
    adaptive_system, adaptive_solver = create_real_time_adaptive_system()
    
    print(f"Created adaptive system with mode: {adaptive_system.mode.value}")
    print(f"Initial parameters: {adaptive_system.param_controller.get_current_settings()}")
    
    # Start adaptation loop
    adaptive_system.start_adaptation_loop()
    
    # Create test problem
    print("\nTesting adaptive solving...")
    G = np.random.rand(10, 10) * 1e-4
    G = G + 0.5 * np.eye(10)  # Well-conditioned
    b = np.random.rand(10)
    
    print(f"Problem characteristics: Size={G.shape}, Condition={np.linalg.cond(G):.2e}")
    
    # Solve adaptively
    solution, iterations, info = adaptive_solver.solve(G, b)
    residual = np.linalg.norm(G @ solution - b)
    
    print(f"Adaptive solution completed:")
    print(f"  Iterations: {iterations}")
    print(f"  Residual: {residual:.2e}")
    print(f"  Execution time: {info.get('execution_time', 'N/A')}s")
    
    # Check updated parameters
    updated_params = adaptive_system.param_controller.get_current_settings()
    print(f"  Updated parameters: {updated_params}")
    
    # Create an adaptive neural network
    print(f"\nCreating adaptive neural network...")
    adaptive_nn = AdaptiveNeuralNetwork(
        adaptive_system,
        architecture=[10, 15, 8, 1],  # [input, hidden1, hidden2, output]
        rram_interface=None  # No specific interface for demo
    )
    
    # Test neural network
    input_signal = np.random.rand(10)
    nn_output = adaptive_nn.forward(input_signal)
    print(f"Neural network output shape: {nn_output.shape}")
    
    # Simulate system over time with changing conditions
    print(f"\nSimulating adaptation over time...")
    for i in range(5):
        # Simulate changing environmental conditions
        env = adaptive_system.environment_sensor.update_environment()
        print(f"  Time step {i+1}: Temperature={env['temperature_k']:.1f}K")
        
        # Update performance metrics to trigger adaptation
        adaptive_system.update_performance_metrics(
            residual_norm=np.random.uniform(1e-6, 1e-3),
            execution_time=np.random.uniform(0.001, 0.01),
            accuracy=np.random.uniform(0.95, 0.99)
        )
        
        time.sleep(0.1)  # Brief delay for demonstration
    
    # Stop adaptation loop
    adaptive_system.stop_adaptation_loop()
    
    # Show adaptation history
    history = adaptive_system.param_controller.adaptation_history
    print(f"\nAdaptation history: {len(history)} changes recorded")
    if history:
        latest_changes = history[-3:]  # Show last 3 changes
        for change in latest_changes:
            print(f"  {change['parameter']}: {change['old_value']:.3f} → {change['new_value']:.3f} ({change['reason']})")
    
    print("\nReal-time adaptive systems demonstration completed!")


if __name__ == "__main__":
    demonstrate_adaptive_systems()