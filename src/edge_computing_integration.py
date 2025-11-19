"""
Edge Computing Integration for RRAM Systems.

This module provides tools for optimizing RRAM-based computations
for edge computing scenarios with power, latency, and resource constraints.
"""
import numpy as np
import time
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import threading
import queue
from dataclasses import dataclass
from enum import Enum

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV
    from .neuromorphic_modules import RRAMNeuromorphicNetwork, RRAMSpikingLayer
    from .optimization_algorithms import RRAMAwareOptimizer
    from .benchmarking_suite import RRAMBenchmarkSuite
    from .advanced_materials_simulation import AdvancedMaterialsSimulator, MaterialSpecificModel
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class EdgeDeviceType(Enum):
    """Types of edge devices."""
    MICROCONTROLLER = "microcontroller"
    SYSTEM_ON_CHIP = "system_on_chip"
    FPGA = "fpga"
    NEUROMORPHIC_CHIP = "neuromorphic_chip"
    RRAM_CHIP = "rram_chip"


class PowerConstraint(Enum):
    """Power constraint levels."""
    ULTRA_LOW = "ultra_low"      # < 1mW
    LOW = "low"                 # 1-10mW
    MEDIUM = "medium"           # 10-100mW
    HIGH = "high"              # > 100mW


class LatencyConstraint(Enum):
    """Latency constraint levels."""
    ULTRA_LOW = "ultra_low"    # < 1ms
    LOW = "low"               # 1-10ms
    MEDIUM = "medium"         # 10-100ms
    HIGH = "high"            # > 100ms


@dataclass
class EdgeConfiguration:
    """Configuration for edge computing optimization."""
    device_type: EdgeDeviceType
    power_limit: float  # in watts
    latency_limit: float  # in seconds
    memory_limit: float  # in MB
    compute_capability: float  # relative compute power (1.0 = baseline)
    temperature_range: Tuple[float, float]  # min/max temperature in Celsius
    operating_voltage: float  # in volts


class EdgeRRAMOptimizer:
    """
    Optimizer for RRAM operations on edge devices with resource constraints.
    """
    
    def __init__(self, 
                 config: EdgeConfiguration,
                 rram_interface: Optional[object] = None):
        """
        Initialize the edge RRAM optimizer.
        
        Args:
            config: Edge device configuration
            rram_interface: Optional RRAM interface
        """
        self.config = config
        self.rram_interface = rram_interface
        self.power_monitor = PowerMonitor()
        self.latency_monitor = LatencyMonitor()
        self.resource_allocator = ResourceAllocator(config)
    
    def optimize_linear_system(self, 
                              G: np.ndarray, 
                              b: np.ndarray,
                              **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize linear system solving for edge constraints.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (solution, metadata)
        """
        start_time = time.time()
        
        # Determine optimal approach based on constraints
        approach = self._select_optimal_approach(G, b)
        
        # Get initial parameters
        params = self._get_constraint_aware_parameters(G, b)
        
        # Apply resource allocation
        allocated_resources = self.resource_allocator.allocate_resources(
            task_type="linear_solver",
            matrix_size=G.shape[0]
        )
        
        # Adjust parameters based on allocated resources
        params.update(allocated_resources)
        
        # Choose solver based on approach
        if approach == "hp_inv":
            solution, iterations, info = hp_inv(G, b, **params)
        elif approach == "block_hp_inv" and G.shape[0] > 16:
            solution, iterations, info = block_hp_inv(G, b, **params)
        elif approach == "adaptive_hp_inv":
            solution, iterations, info = adaptive_hp_inv(G, b, **params)
        else:
            # Fallback to standard approach
            solution, iterations, info = hp_inv(G, b, **params)
        
        execution_time = time.time() - start_time
        
        # Monitor resource usage
        power_consumption = self.power_monitor.estimate_power(
            operations=len(G)**3,  # Rough estimate
            precision=params.get('bits', 4)
        )
        
        metadata = {
            'approach': approach,
            'execution_time': execution_time,
            'power_consumption': power_consumption,
            'allocated_resources': allocated_resources,
            'matrix_properties': {
                'size': G.shape[0],
                'condition_number': float(np.linalg.cond(G)),
                'sparsity': float(np.count_nonzero(G) / G.size)
            },
            'solution_iterations': iterations
        }
        
        return solution, metadata
    
    def _select_optimal_approach(self, G: np.ndarray, b: np.ndarray) -> str:
        """
        Select the optimal approach based on edge constraints.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            
        Returns:
            Selected approach name
        """
        matrix_size = G.shape[0]
        cond_num = np.linalg.cond(G)
        
        # Evaluate different factors
        is_power_critical = self.config.power_limit < 0.01  # Less than 10mW
        is_latency_critical = self.config.latency_limit < 0.01  # Less than 10ms
        is_memory_limited = self.config.memory_limit < 10  # Less than 10MB
        
        # Select approach based on constraints
        if matrix_size < 10 and is_latency_critical:
            return "hp_inv"  # Direct approach for small problems
        elif matrix_size > 50 and not is_memory_limited:
            return "block_hp_inv"  # Block approach for larger problems
        elif cond_num > 1e8:  # Ill-conditioned
            return "adaptive_hp_inv"  # Adaptive for difficult problems
        else:
            return "hp_inv"  # Default approach
    
    def _get_constraint_aware_parameters(self, G: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """
        Get RRAM parameters that respect edge constraints.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            
        Returns:
            Parameter dictionary
        """
        params = {}
        
        # Determine precision based on power and accuracy requirements
        if self.config.power_limit < 0.005:  # Very low power
            params['bits'] = 2  # Minimum precision
        elif self.config.power_limit < 0.02:  # Low power
            params['bits'] = 3
        else:  # Higher power budget allows more precision
            params['bits'] = 5
        
        # Adjust noise based on precision
        params['lp_noise_std'] = min(0.1, 0.01 * (5 - params['bits']))
        
        # Iteration limits based on latency constraints
        if self.config.latency_limit < 0.001:  # 1ms limit
            params['max_iter'] = 5
        elif self.config.latency_limit < 0.01:  # 10ms limit
            params['max_iter'] = 10
        else:  # More time available
            params['max_iter'] = 20
        
        # Relaxation factor based on stability requirements
        cond_num = np.linalg.cond(G)
        if cond_num > 1e10:
            params['relaxation_factor'] = 0.5  # More conservative for ill-conditioned
        elif cond_num > 1e6:
            params['relaxation_factor'] = 0.7
        else:
            params['relaxation_factor'] = 1.0
        
        # Tolerance based on required accuracy
        params['tol'] = 1e-4
        
        return params


class PowerMonitor:
    """
    Monitor and estimate power consumption for RRAM operations.
    """
    
    def __init__(self):
        self.cumulative_power = 0.0
        self.power_log = []
    
    def estimate_power(self, 
                      operations: int, 
                      precision: int = 4,
                      temperature: float = 300.0) -> float:
        """
        Estimate power consumption for operations.
        
        Args:
            operations: Number of operations
            precision: Bit precision (affects power)
            temperature: Operating temperature in Kelvin
            
        Returns:
            Estimated power consumption in watts
        """
        # Base power estimate per operation (in W)
        base_power_per_op = 1e-9  # 1 nW per operation
        
        # Scale by precision (more bits = more power)
        precision_factor = precision / 4.0  # Normalize to 4-bit baseline
        
        # Temperature effects on power
        temp_factor = 1.0 + 0.001 * (temperature - 300.0)  # 0.1% per degree
        
        estimated_power = operations * base_power_per_op * precision_factor * temp_factor
        
        # Log the estimation
        self.power_log.append({
            'timestamp': time.time(),
            'operations': operations,
            'precision': precision,
            'temperature': temperature,
            'estimated_power': estimated_power
        })
        
        # Limit log size
        if len(self.power_log) > 1000:
            self.power_log = self.power_log[-500:]
        
        return estimated_power
    
    def get_average_power(self) -> float:
        """Get average power consumption from the log."""
        if not self.power_log:
            return 0.0
        
        total_power = sum(entry['estimated_power'] for entry in self.power_log)
        return total_power / len(self.power_log)


class LatencyMonitor:
    """
    Monitor and predict latency for RRAM operations.
    """
    
    def __init__(self):
        self.latency_log = []
        self.predictive_model = {}  # Simple historical model
    
    def estimate_latency(self,
                        matrix_size: int,
                        precision: int = 4,
                        operations: str = "solve") -> float:
        """
        Estimate operation latency.
        
        Args:
            matrix_size: Size of the matrix operation
            precision: Bit precision
            operations: Type of operation ("solve", "multiply", "invert")
            
        Returns:
            Estimated latency in seconds
        """
        # Base latencies (in seconds) for different operations
        base_latencies = {
            "solve": 1e-6,      # 1 µs for solving
            "multiply": 1e-7,   # 0.1 µs for multiplication
            "invert": 5e-6      # 5 µs for inversion
        }
        
        base_latency = base_latencies.get(operations, 1e-6)
        
        # Scale by matrix size (typically O(n²) to O(n³))
        size_factor = matrix_size ** 2.5  # Conservative estimate
        
        # Scale by precision (more bits = more time)
        precision_factor = precision / 4.0  # Normalize to 4-bit baseline
        
        estimated_latency = base_latency * size_factor * precision_factor
        
        # Add to log
        self.latency_log.append({
            'timestamp': time.time(),
            'matrix_size': matrix_size,
            'precision': precision,
            'operation': operations,
            'estimated_latency': estimated_latency
        })
        
        # Limit log size
        if len(self.latency_log) > 1000:
            self.latency_log = self.latency_log[-500:]
        
        return estimated_latency
    
    def get_predicted_latency(self, 
                            matrix_size: int,
                            operation: str = "solve") -> float:
        """
        Get predicted latency based on historical data.
        
        Args:
            matrix_size: Size of the matrix operation
            operation: Type of operation
            
        Returns:
            Predicted latency in seconds
        """
        # Filter log for same operation
        relevant_entries = [
            entry for entry in self.latency_log 
            if entry['operation'] == operation
        ]
        
        if not relevant_entries:
            # Fall back to estimate
            return self.estimate_latency(matrix_size, operations=operation)
        
        # Calculate size-normalized latency
        normalized_latencies = [
            entry['estimated_latency'] / (entry['matrix_size'] ** 2.5)
            for entry in relevant_entries
        ]
        
        avg_normalized = np.mean(normalized_latencies)
        predicted = avg_normalized * (matrix_size ** 2.5)
        
        return predicted


class ResourceAllocator:
    """
    Allocate resources based on device constraints.
    """
    
    def __init__(self, config: EdgeConfiguration):
        self.config = config
    
    def allocate_resources(self, 
                          task_type: str,
                          matrix_size: int) -> Dict[str, Any]:
        """
        Allocate resources based on task and constraints.
        
        Args:
            task_type: Type of task ('linear_solver', 'neural_network', etc.)
            matrix_size: Size of the primary operation
            
        Returns:
            Resource allocation dictionary
        """
        allocation = {}
        
        # Calculate available resources
        available_time = self.config.latency_limit
        available_power = self.config.power_limit
        available_memory = self.config.memory_limit * 1024 * 1024  # Convert to bytes
        
        # Resource allocation based on task
        if task_type == "linear_solver":
            # For linear solvers, precision and iteration count are key
            if available_power < 0.01:  # Very low power
                allocation['bits'] = 2
                allocation['max_iter'] = min(10, int(available_time * 100))  # 100 it/s assumption
            elif available_power < 0.1:  # Low to medium power
                allocation['bits'] = min(6, max(3, int(available_power * 100)))
                allocation['max_iter'] = min(20, int(available_time * 200))  # 200 it/s assumption
            else:  # Higher power budget
                allocation['bits'] = 6
                allocation['max_iter'] = min(50, int(available_time * 1000))  # 1000 it/s assumption
            
            # Adjust for matrix size (larger problems may need fewer iterations to stay within time)
            size_adjustment = min(1.0, 100.0 / matrix_size)  # Reduce for larger problems
            allocation['max_iter'] = int(allocation['max_iter'] * size_adjustment)
            
        elif task_type == "neural_network":
            # For neural networks, memory and precision are key
            allocation['quantization_bits'] = min(8, max(2, int(available_power * 50)))
            allocation['network_depth'] = min(10, int(available_memory / (matrix_size * 1000)))
        
        return allocation


class EdgeRRAMInferenceEngine:
    """
    Inference engine optimized for edge RRAM deployments.
    """
    
    def __init__(self,
                 rram_interface: Optional[object] = None,
                 config: Optional[EdgeConfiguration] = None):
        """
        Initialize the edge inference engine.
        
        Args:
            rram_interface: Optional RRAM interface
            config: Edge device configuration
        """
        self.rram_interface = rram_interface
        self.config = config or EdgeConfiguration(
            device_type=EdgeDeviceType.MICROCONTROLLER,
            power_limit=0.01,  # 10mW
            latency_limit=0.05,  # 50ms
            memory_limit=1.0,    # 1MB
            compute_capability=1.0,
            temperature_range=(0, 70),  # 0-70°C
            operating_voltage=3.3
        )
        
        self.optimizer = EdgeRRAMOptimizer(self.config, self.rram_interface)
        self.power_monitor = PowerMonitor()
        self.latency_monitor = LatencyMonitor()
        self.models_cache = {}
    
    def infer(self, 
              model_id: str,
              input_data: np.ndarray,
              **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform inference optimized for edge constraints.
        
        Args:
            model_id: ID of the model to use
            input_data: Input data for inference
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (output, metadata)
        """
        start_time = time.time()
        
        # Load model if not in cache
        if model_id not in self.models_cache:
            model = self._load_model(model_id)
            self.models_cache[model_id] = model
        else:
            model = self.models_cache[model_id]
        
        # Optimize for edge constraints
        output, inference_metadata = self._perform_edge_inference(model, input_data, **kwargs)
        
        execution_time = time.time() - start_time
        
        # Calculate resource usage
        power_consumption = self.power_monitor.estimate_power(
            operations=input_data.size,
            precision=kwargs.get('quantization_bits', 4)
        )
        
        metadata = {
            'execution_time': execution_time,
            'power_consumption': power_consumption,
            'model_id': model_id,
            'input_shape': input_data.shape,
            'output_shape': output.shape,
            'inference_metadata': inference_metadata
        }
        
        return output, metadata
    
    def _load_model(self, model_id: str):
        """
        Load a model for inference.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Loaded model
        """
        # This is a simplified model loading
        # In practice, this would load from storage or construct based on ID
        return {
            'model_id': model_id,
            'parameters': np.random.rand(10, 10)  # Dummy parameters
        }
    
    def _perform_edge_inference(self, 
                               model: Dict[str, Any], 
                               input_data: np.ndarray,
                               **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform actual inference with edge optimizations.
        
        Args:
            model: Model to use
            input_data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (output, inference metadata)
        """
        # Apply input quantization if needed
        quantization_bits = kwargs.get('quantization_bits', 4)
        if quantization_bits < 8:
            # Quantize input
            input_range = np.max(input_data) - np.min(input_data)
            levels = 2**quantization_bits - 1
            scale = levels / input_range if input_range != 0 else 1
            quantized_input = np.round((input_data - np.min(input_data)) * scale)
            quantized_input = quantized_input / scale + np.min(input_data)
        else:
            quantized_input = input_data
        
        # Perform computation using RRAM model
        if self.rram_interface is not None:
            # Use real hardware if available
            try:
                self.rram_interface.write_matrix(model['parameters'])
                output = self.rram_interface.matrix_vector_multiply(quantized_input)
            except Exception:
                # Fallback to simulation if hardware fails
                output = model['parameters'] @ quantized_input
        else:
            # Use simulation
            output = model['parameters'] @ quantized_input
        
        # Apply edge-specific optimizations
        inference_metadata = {
            'quantization_bits_used': quantization_bits,
            'input_quantized': quantization_bits < 8,
            'operations_performed': input_data.size,
            'estimated_latency': self.latency_monitor.estimate_latency(
                input_data.size, quantization_bits, 'multiply'
            )
        }
        
        return output, inference_metadata
    
    def batch_infer(self,
                   model_id: str,
                   input_batch: List[np.ndarray],
                   batch_size: Optional[int] = None) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Perform batch inference optimized for edge devices.
        
        Args:
            model_id: ID of the model to use
            input_batch: List of input arrays
            batch_size: Size of batches (optional)
            
        Returns:
            List of (output, metadata) tuples
        """
        if batch_size is None:
            # Estimate optimal batch size based on memory constraints
            sample_size = input_batch[0].size if input_batch else 1
            # Assume 4 bytes per float; adjust based on memory constraint
            max_batch_elements = int(self.config.memory_limit * 1024 * 1024 / 4)
            batch_size = min(32, max(1, max_batch_elements // sample_size))
        
        results = []
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i+batch_size]
            
            for input_data in batch:
                output, metadata = self.infer(model_id, input_data)
                results.append((output, metadata))
                
                # Check if we're approaching resource limits
                if (metadata['execution_time'] > self.config.latency_limit * 0.8 or
                    metadata['power_consumption'] > self.config.power_limit * 0.8):
                    # Throttle if approaching limits
                    time.sleep(0.001)  # 1ms break
        
        return results


class EdgeRRAMManagementSystem:
    """
    Management system for RRAM operations in edge environments.
    """
    
    def __init__(self, config: EdgeConfiguration):
        """
        Initialize the edge management system.
        
        Args:
            config: Edge device configuration
        """
        self.config = config
        self.active_inferences = 0
        self.resource_usage = {
            'power': 0.0,
            'memory': 0.0,
            'compute': 0.0
        }
        self.temperature_controller = TemperatureController(
            config.temperature_range
        )
        self.task_queue = queue.Queue()
        self.scheduler = EdgeTaskScheduler()
    
    def schedule_inference(self, 
                          model_id: str,
                          input_data: np.ndarray,
                          priority: int = 1) -> str:
        """
        Schedule an inference task with priority.
        
        Args:
            model_id: ID of the model to use
            input_data: Input data
            priority: Task priority (1=high, 2=medium, 3=low)
            
        Returns:
            Task ID
        """
        task_id = f"task_{int(time.time() * 1000000)}"
        task = {
            'id': task_id,
            'model_id': model_id,
            'input_data': input_data,
            'priority': priority,
            'timestamp': time.time()
        }
        
        self.task_queue.put((priority, task))
        
        # Start processing if not already running
        if self.active_inferences == 0:
            self._start_processing()
        
        return task_id
    
    def _start_processing(self):
        """
        Start processing tasks in a background thread.
        """
        def process_tasks():
            while not self.task_queue.empty():
                try:
                    priority, task = self.task_queue.get_nowait()
                    
                    # Check resource availability
                    if not self._check_resource_availability():
                        # Re-queue for later
                        time.sleep(0.01)  # Wait 10ms
                        self.task_queue.put((priority, task))
                        continue
                    
                    # Perform inference
                    engine = EdgeRRAMInferenceEngine(config=self.config)
                    output, metadata = engine.infer(
                        task['model_id'], 
                        task['input_data']
                    )
                    
                    # Update resource usage
                    self._update_resource_usage(metadata)
                    
                    # Process completed task
                    self._handle_completed_task(task, output, metadata)
                    
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error processing task: {e}")
        
        # Start in background
        thread = threading.Thread(target=process_tasks)
        thread.daemon = True
        thread.start()
        self.active_inferences += 1
    
    def _check_resource_availability(self) -> bool:
        """
        Check if resources are available for new tasks.
        
        Returns:
            True if resources are available, False otherwise
        """
        # Check if we're within limits
        return (
            self.resource_usage['power'] < self.config.power_limit * 0.9 and
            self.resource_usage['memory'] < self.config.memory_limit * 0.9 and
            self.resource_usage['compute'] < self.config.compute_capability * 0.9
        )
    
    def _update_resource_usage(self, metadata: Dict[str, Any]):
        """
        Update resource usage based on task metadata.
        
        Args:
            metadata: Task execution metadata
        """
        self.resource_usage['power'] = min(
            self.config.power_limit,
            self.resource_usage['power'] + metadata.get('power_consumption', 0)
        )
        
        # Reset usage periodically
        if time.time() % 1.0 < 0.01:  # Every second
            self.resource_usage['power'] *= 0.8  # Decay usage
    
    def _handle_completed_task(self, task: Dict, output: np.ndarray, metadata: Dict):
        """
        Handle a completed task (could save results, etc.).
        
        Args:
            task: Original task definition
            output: Inference output
            metadata: Execution metadata
        """
        # In a real system, this would handle the results
        # (save to storage, send to next stage, etc.)
        pass


class TemperatureController:
    """
    Control temperature in edge RRAM systems.
    """
    
    def __init__(self, 
                 temperature_range: Tuple[float, float] = (0, 70),
                 thermal_time_constant: float = 10.0):
        """
        Initialize temperature controller.
        
        Args:
            temperature_range: Acceptable temperature range (Celsius)
            thermal_time_constant: Time constant for thermal response (seconds)
        """
        self.min_temp, self.max_temp = temperature_range
        self.current_temp = 25.0  # Start at room temperature
        self.thermal_time_constant = thermal_time_constant
        self.thermal_mass = 1.0  # Simplified thermal mass
        self.heat_sources = []
    
    def update_temperature(self, 
                          power_generation: float,
                          ambient_temp: float = 25.0,
                          time_step: float = 1.0) -> float:
        """
        Update system temperature based on power generation.
        
        Args:
            power_generation: Power generated (W)
            ambient_temp: Ambient temperature (C)
            time_step: Time step for simulation (s)
            
        Returns:
            Updated temperature (C)
        """
        # Calculate temperature change
        heat_generated = power_generation * time_step
        heat_loss = (self.current_temp - ambient_temp) / self.thermal_time_constant
        
        temp_change = (heat_generated - heat_loss) / self.thermal_mass
        
        self.current_temp += temp_change
        
        # Apply bounds
        self.current_temp = np.clip(self.current_temp, self.min_temp, self.max_temp)
        
        return self.current_temp
    
    def is_temperature_safe(self) -> bool:
        """
        Check if temperature is within safe range.
        
        Returns:
            True if temperature is safe, False otherwise
        """
        return self.min_temp <= self.current_temp <= self.max_temp
    
    def get_thermal_derating_factor(self) -> float:
        """
        Get performance derating factor based on temperature.
        
        Returns:
            Derating factor (0.0 to 1.0)
        """
        # Performance degrades as temperature approaches limits
        if self.current_temp <= self.min_temp + 5:
            return 0.9  # Performance reduced at low temp
        elif self.current_temp >= self.max_temp - 5:
            return 0.7  # Significant derating near max temp
        else:
            # Linear derating near limits
            margin = min(
                (self.current_temp - self.min_temp) / (self.max_temp - self.min_temp),
                (self.max_temp - self.current_temp) / (self.max_temp - self.min_temp)
            )
            return 0.8 + 0.2 * margin


class EdgeTaskScheduler:
    """
    Schedule tasks based on edge constraints.
    """
    
    def __init__(self):
        self.task_queue = []
    
    def add_task(self, task: Dict, priority: int = 2):
        """
        Add a task to the scheduler.
        
        Args:
            task: Task definition
            priority: Priority level (1=high, 2=medium, 3=low)
        """
        self.task_queue.append((priority, task))
        # Sort by priority (lower number = higher priority)
        self.task_queue.sort(key=lambda x: x[0])
    
    def get_next_task(self) -> Optional[Tuple[int, Dict]]:
        """
        Get the next highest priority task.
        
        Returns:
            Tuple of (priority, task) or None if no tasks
        """
        return self.task_queue.pop(0) if self.task_queue else None


def create_edge_optimized_system(config: EdgeConfiguration) -> Tuple[EdgeRRAMInferenceEngine, EdgeRRAMManagementSystem]:
    """
    Create an edge-optimized RRAM system.
    
    Args:
        config: Edge device configuration
        
    Returns:
        Tuple of (inference engine, management system)
    """
    engine = EdgeRRAMInferenceEngine(config=config)
    management = EdgeRRAMManagementSystem(config)
    
    return engine, management


def demonstrate_edge_integration():
    """
    Demonstrate the edge computing integration capabilities.
    """
    print("Edge Computing Integration Demonstration")
    print("="*50)
    
    # Create edge configuration
    config = EdgeConfiguration(
        device_type=EdgeDeviceType.MICROCONTROLLER,
        power_limit=0.05,  # 50mW
        latency_limit=0.02,  # 20ms
        memory_limit=2.0,    # 2MB
        compute_capability=0.5,  # Half the baseline
        temperature_range=(0, 85),  # Wide temp range
        operating_voltage=3.3
    )
    
    print(f"Created edge config: Power={config.power_limit}W, Latency={config.latency_limit}s")
    
    # Create edge-optimized system
    engine, management = create_edge_optimized_system(config)
    print("Created edge-optimized inference engine and management system")
    
    # Create sample data for inference
    sample_data = np.random.rand(10)  # Small vector for microcontroller
    
    # Perform edge-optimized inference
    output, metadata = engine.infer("model_1", sample_data, quantization_bits=4)
    print(f"Edge inference completed. Output shape: {output.shape}")
    print(f"Execution time: {metadata['execution_time']:.4f}s")
    print(f"Power consumption: {metadata['power_consumption']:.6f}W")
    
    # Test linear system solving with edge constraints
    optimizer = EdgeRRAMOptimizer(config)
    
    G = np.random.rand(8, 8) * 1e-4
    G = G + 0.5 * np.eye(8)  # Well-conditioned
    b = np.random.rand(8)
    
    solution, solve_metadata = optimizer.optimize_linear_system(G, b)
    print(f"Edge-optimized linear solve completed")
    print(f"Approach: {solve_metadata['approach']}")
    print(f"Solution iterations: {solve_metadata['solution_iterations']}")
    print(f"Matrix condition: {solve_metadata['matrix_properties']['condition_number']:.2e}")
    
    # Test temperature controller
    temp_controller = TemperatureController(config.temperature_range)
    new_temp = temp_controller.update_temperature(power_generation=0.01, time_step=5.0)
    print(f"Temperature after 5s with 10mW: {new_temp:.1f}°C")
    print(f"Temperature safe: {temp_controller.is_temperature_safe()}")
    print(f"Thermal derating: {temp_controller.get_thermal_derating_factor():.2f}")
    
    print("\nEdge computing integration demonstration completed!")


if __name__ == "__main__":
    demonstrate_edge_integration()