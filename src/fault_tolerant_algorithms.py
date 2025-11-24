"""
Fault-Tolerant Algorithms for RRAM-based Computing Systems.

This module provides algorithms that can work around RRAM defects,
degradation, and other reliability issues by implementing various
fault-tolerant techniques.
"""
import numpy as np
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import time
from enum import Enum
from dataclasses import dataclass

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV
    from .neuromorphic_modules import RRAMSpikingLayer, RRAMReservoirComputer
    from .optimization_algorithms import RRAMAwareOptimizer
    from .redundancy import apply_redundancy
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class FaultType(Enum):
    """Types of faults in RRAM systems."""
    STUCK_AT_FAULT = "stuck_at_fault"
    DRIFT = "drift"
    VARIABILITY = "variability"
    TDDB = "tddb"  # Time Dependent Dielectric Breakdown
    THERMAL = "thermal"
    AGING = "aging"


class FaultToleranceStrategy(Enum):
    """Strategies for handling faults."""
    REDUNDANCY = "redundancy"
    RETRY = "retry"
    ADAPTIVE = "adaptive"
    MARGIN = "margin"
    ERROR_CORRECTION = "error_correction"


@dataclass
class FaultToleranceResult:
    """Result of fault-tolerant computation."""
    solution: np.ndarray
    success: bool
    recovery_attempts: int
    fault_detected: bool
    fault_type: Optional[FaultType]
    recovery_strategy: FaultToleranceStrategy
    execution_time: float
    error_metrics: Dict[str, float]


class FaultDetector:
    """
    Detects various types of faults in RRAM systems.
    """
    
    def __init__(self, 
                 variability_threshold: float = 0.1,
                 drift_threshold: float = 0.05,
                 stuck_fault_threshold: float = 1e-10):
        """
        Initialize fault detector.
        
        Args:
            variability_threshold: Threshold for variability detection
            drift_threshold: Threshold for drift detection
            stuck_fault_threshold: Threshold for stuck-at-fault detection
        """
        self.variability_threshold = variability_threshold
        self.drift_threshold = drift_threshold
        self.stuck_fault_threshold = stuck_fault_threshold
    
    def detect_stuck_at_faults(self, matrix: np.ndarray) -> np.ndarray:
        """
        Detect stuck-at-faults in a conductance matrix.
        
        Args:
            matrix: Conductance matrix to analyze
            
        Returns:
            Boolean matrix indicating stuck-at-fault locations
        """
        # Stuck at low conductance
        stuck_low = matrix < self.stuck_fault_threshold
        # Stuck at high conductance (could be modeled differently)
        stuck_high = matrix > 1e-3  # Very high conductance could indicate other issues
        
        return np.logical_or(stuck_low, stuck_high)
    
    def detect_variability(self, matrix: np.ndarray) -> np.ndarray:
        """
        Detect high variability in conductance values.
        
        Args:
            matrix: Conductance matrix to analyze
            
        Returns:
            Boolean matrix indicating high variability locations
        """
        # Calculate local statistics to detect outliers
        mean_conductance = np.mean(matrix)
        std_conductance = np.std(matrix)
        
        if std_conductance == 0:
            return np.zeros_like(matrix, dtype=bool)
        
        # Values that are more than threshold standard deviations from mean
        z_scores = np.abs((matrix - mean_conductance) / std_conductance)
        high_variability = z_scores > self.variability_threshold * 10  # Adjust threshold
        
        return high_variability
    
    def detect_drift(self, 
                    old_matrix: np.ndarray, 
                    new_matrix: np.ndarray, 
                    time_elapsed: float) -> np.ndarray:
        """
        Detect drift by comparing old and new conductance values.
        
        Args:
            old_matrix: Previous conductance matrix
            new_matrix: Current conductance matrix
            time_elapsed: Time elapsed between measurements
            
        Returns:
            Boolean matrix indicating drift locations
        """
        if time_elapsed <= 0:
            return np.zeros_like(new_matrix, dtype=bool)
        
        # Calculate relative change per unit time
        relative_change = np.abs(new_matrix - old_matrix) / (old_matrix + 1e-12)
        drift_rate = relative_change / time_elapsed
        
        # Identify locations with high drift rate
        significant_drift = drift_rate > self.drift_threshold
        
        return significant_drift
    
    def analyze_matrix_health(self, matrix: np.ndarray) -> Dict[FaultType, np.ndarray]:
        """
        Analyze a matrix for different types of faults.
        
        Args:
            matrix: Conductance matrix to analyze
            
        Returns:
            Dictionary mapping fault types to boolean matrices
        """
        fault_map = {}
        
        # Detect stuck-at faults
        fault_map[FaultType.STUCK_AT_FAULT] = self.detect_stuck_at_faults(matrix)
        
        # Detect variability issues
        fault_map[FaultType.VARIABILITY] = self.detect_variability(matrix)
        
        return fault_map


class FaultTolerantHPINV:
    """
    Fault-tolerant implementation of HP-INV algorithm.
    """
    
    def __init__(self,
                 rram_interface: Optional[object] = None,
                 max_retries: int = 3,
                 fault_tolerance_strategy: FaultToleranceStrategy = FaultToleranceStrategy.REDUNDANCY,
                 confidence_threshold: float = 0.95):
        """
        Initialize fault-tolerant HP-INV.
        
        Args:
            rram_interface: Optional RRAM interface for hardware feedback
            max_retries: Maximum number of retry attempts
            fault_tolerance_strategy: Strategy for handling faults
            confidence_threshold: Threshold for solution confidence
        """
        self.rram_interface = rram_interface
        self.max_retries = max_retries
        self.fault_tolerance_strategy = fault_tolerance_strategy
        self.confidence_threshold = confidence_threshold
        self.fault_detector = FaultDetector()
    
    def solve(self, 
              G: np.ndarray, 
              b: np.ndarray, 
              **kwargs) -> FaultToleranceResult:
        """
        Solve Gx = b using fault-tolerant HP-INV.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            FaultToleranceResult with solution and metadata
        """
        start_time = time.time()
        
        # Detect faults in the initial matrix
        fault_maps = self.fault_detector.analyze_matrix_health(G)
        total_faults = sum(np.sum(mask) for mask in fault_maps.values())
        fault_detected = total_faults > 0
        
        recovery_attempts = 0
        solution = None
        success = False
        fault_type = None
        
        # Apply fault tolerance based on detected faults
        if fault_detected:
            if self.fault_tolerance_strategy == FaultToleranceStrategy.REDUNDANCY:
                # Apply redundancy to repair faults
                G_repaired = self._apply_redundancy(G, fault_maps)
                solution, iterations, info = hp_inv(G_repaired, b, **kwargs)
                success = True
                
            elif self.fault_tolerance_strategy == FaultToleranceStrategy.ADAPTIVE:
                # Use adaptive algorithm that adjusts parameters based on faults
                solution = self._solve_with_adaptation(G, b, fault_maps, **kwargs)
                success = solution is not None
                
            elif self.fault_tolerance_strategy == FaultToleranceStrategy.RETRY:
                # Attempt solution with retries
                solution, success, recovery_attempts = self._solve_with_retries(G, b, **kwargs)
                
            elif self.fault_tolerance_strategy == FaultToleranceStrategy.MARGIN:
                # Use safety margins to account for potential faults
                solution = self._solve_with_margins(G, b, fault_maps, **kwargs)
                success = solution is not None
        else:
            # No faults detected, use standard solver
            try:
                solution, iterations, info = hp_inv(G, b, **kwargs)
                success = True
            except Exception:
                # If standard method fails, try with tolerance
                solution, iterations, info = hp_inv(G, b, lp_noise_std=0.02, **kwargs)
                success = True
        
        # Validate solution if we got one
        if solution is not None:
            residual = np.linalg.norm(G @ solution - b)
            error_metrics = {
                'residual_norm': float(residual),
                'relative_error': float(residual / (np.linalg.norm(b) + 1e-12)),
                'condition_number': float(np.linalg.cond(G) if np.linalg.cond(G) < 1e15 else 1e15)
            }
            
            # Determine if solution is acceptable
            if error_metrics['residual_norm'] > 1e-3 or error_metrics['condition_number'] > 1e12:
                success = False
        else:
            error_metrics = {'residual_norm': float('inf'), 'relative_error': float('inf')}
        
        execution_time = time.time() - start_time
        
        return FaultToleranceResult(
            solution=solution if solution is not None else np.zeros_like(b),
            success=success,
            recovery_attempts=recovery_attempts,
            fault_detected=fault_detected,
            fault_type=list(fault_maps.keys())[0] if fault_maps else None,
            recovery_strategy=self.fault_tolerance_strategy,
            execution_time=execution_time,
            error_metrics=error_metrics
        )
    
    def _apply_redundancy(self, 
                         G: np.ndarray, 
                         fault_maps: Dict[FaultType, np.ndarray]) -> np.ndarray:
        """
        Apply redundancy to repair faults in the matrix.
        
        Args:
            G: Original matrix
            fault_maps: Maps of detected faults
            
        Returns:
            Matrix with faults repaired through redundancy
        """
        # Combine all fault locations
        combined_faults = np.zeros_like(G, dtype=bool)
        for fault_mask in fault_maps.values():
            combined_faults = np.logical_or(combined_faults, fault_mask)
        
        # Apply redundancy repair
        G_repaired = G.copy()
        n = G.shape[0]
        
        # For each fault location, try to repair using neighboring values or mathematical relationships
        for i in range(n):
            for j in range(n):
                if combined_faults[i, j]:
                    # Find a suitable replacement value
                    # Option 1: Use average of neighbors
                    neighbors = []
                    if i > 0: neighbors.append(G[i-1, j])
                    if i < n-1: neighbors.append(G[i+1, j])
                    if j > 0: neighbors.append(G[i, j-1])
                    if j < n-1: neighbors.append(G[i, j+1])
                    
                    if neighbors:
                        G_repaired[i, j] = np.mean(neighbors)
                    else:
                        # If no good neighbors, use matrix structure information
                        G_repaired[i, j] = np.mean(G)  # Fallback to average
        
        return G_repaired
    
    def _solve_with_adaptation(self, 
                              G: np.ndarray, 
                              b: np.ndarray, 
                              fault_maps: Dict[FaultType, np.ndarray],
                              **kwargs) -> Optional[np.ndarray]:
        """
        Solve using adaptive parameters based on detected faults.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            fault_maps: Maps of detected faults
            **kwargs: Additional parameters
            
        Returns:
            Solution vector or None if failed
        """
        # Adjust parameters based on fault characteristics
        params = kwargs.copy()
        
        # Count total faults to adjust parameters
        total_faults = sum(np.sum(mask) for mask in fault_maps.values())
        fault_ratio = total_faults / G.size
        
        # Increase noise tolerance for noisier conditions
        params['lp_noise_std'] = params.get('lp_noise_std', 0.01) * (1 + fault_ratio * 10)
        
        # Increase iterations for more difficult problems
        params['max_iter'] = int(params.get('max_iter', 10) * (1 + fault_ratio * 5))
        
        # Adjust relaxation for stability
        params['relaxation_factor'] = min(params.get('relaxation_factor', 0.9), 0.9)
        
        try:
            solution, iterations, info = hp_inv(G, b, **params)
            return solution
        except Exception:
            return None
    
    def _solve_with_retries(self, 
                           G: np.ndarray, 
                           b: np.ndarray, 
                           **kwargs) -> Tuple[Optional[np.ndarray], bool, int]:
        """
        Solve with retry mechanism.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (solution, success flag, number of attempts)
        """
        for attempt in range(self.max_retries):
            try:
                # Slightly perturb the matrix to try different paths
                if attempt > 0:
                    G_perturbed = G + np.random.normal(0, 1e-6, G.shape)
                else:
                    G_perturbed = G
                
                solution, iterations, info = hp_inv(G_perturbed, b, **kwargs)
                
                # Validate solution
                residual = np.linalg.norm(G @ solution - b)
                if residual < 1e-3:  # Good enough
                    return solution, True, attempt + 1
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    return None, False, attempt + 1
        
        return None, False, self.max_retries
    
    def _solve_with_margins(self, 
                           G: np.ndarray, 
                           b: np.ndarray, 
                           fault_maps: Dict[FaultType, np.ndarray],
                           **kwargs) -> Optional[np.ndarray]:
        """
        Solve with safety margins to account for potential faults.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            fault_maps: Maps of detected faults
            **kwargs: Additional parameters
            
        Returns:
            Solution vector or None if failed
        """
        total_faults = sum(np.sum(mask) for mask in fault_maps.values())
        fault_ratio = total_faults / G.size
        
        # Use more conservative parameters
        params = kwargs.copy()
        params['tol'] = params.get('tol', 1e-6) * (1 + fault_ratio * 100)  # Looser tolerance
        params['max_iter'] = int(params.get('max_iter', 10) * (1 + fault_ratio * 20))  # More iterations
        
        try:
            solution, iterations, info = hp_inv(G, b, **params)
            return solution
        except Exception:
            # If HP-INV fails, try with reduced precision or different algorithm
            try:
                # Use pseudo-inverse as fallback
                solution = np.linalg.lstsq(G, b, rcond=None)[0]
                return solution
            except (np.linalg.LinAlgError, ValueError) as e:
                warnings.warn(f"Failed to solve system with HP-INV and lstsq fallback: {e}")
                return None


class AdaptiveFaultTolerance:
    """
    Adaptive fault tolerance that learns from system behavior.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 learning_rate: float = 0.1):
        """
        Initialize adaptive fault tolerance system.
        
        Args:
            rram_interface: Optional RRAM interface
            learning_rate: Learning rate for adaptation
        """
        self.rram_interface = rram_interface
        self.learning_rate = learning_rate
        self.fault_history = []
        self.performance_history = []
        
        # Strategy effectiveness weights
        self.strategy_weights = {
            FaultToleranceStrategy.REDUNDANCY: 1.0,
            FaultToleranceStrategy.RETRY: 1.0,
            FaultToleranceStrategy.ADAPTIVE: 1.0,
            FaultToleranceStrategy.MARGIN: 1.0,
            FaultToleranceStrategy.ERROR_CORRECTION: 1.0
        }
    
    def select_best_strategy(self, 
                           fault_info: Dict[FaultType, np.ndarray],
                           matrix_properties: Dict[str, float]) -> FaultToleranceStrategy:
        """
        Select the best fault tolerance strategy based on current conditions.
        
        Args:
            fault_info: Information about detected faults
            matrix_properties: Properties of the matrix being solved
            
        Returns:
            Best strategy for current conditions
        """
        # Calculate fault characteristics
        total_faults = sum(np.sum(mask) for mask in fault_info.values())
        fault_density = total_faults / np.prod(list(fault_info.values())[0].shape) if fault_info else 0
        
        # Matrix condition
        cond_num = matrix_properties.get('condition_number', 1.0)
        ill_conditioned = cond_num > 1e10
        
        # Decide based on fault type and density
        if fault_density < 0.01:  # Very few faults
            return FaultToleranceStrategy.ADAPTIVE
        elif fault_density < 0.05:  # Moderate faults
            if ill_conditioned:
                return FaultToleranceStrategy.MARGIN
            else:
                return FaultToleranceStrategy.REDUNDANCY
        else:  # Many faults
            return FaultToleranceStrategy.RETRY
    
    def solve_adaptively(self, 
                        G: np.ndarray, 
                        b: np.ndarray, 
                        **kwargs) -> FaultToleranceResult:
        """
        Solve adaptively based on system behavior.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            FaultToleranceResult with solution and metadata
        """
        # Detect faults
        detector = FaultDetector()
        fault_info = detector.analyze_matrix_health(G)
        
        # Get matrix properties
        matrix_properties = {
            'condition_number': float(np.linalg.cond(G)),
            'size': G.shape[0],
            'sparsity': float(np.count_nonzero(G) / G.size)
        }
        
        # Select best strategy
        best_strategy = self.select_best_strategy(fault_info, matrix_properties)
        
        # Create fault-tolerant solver with selected strategy
        ft_hpiv = FaultTolerantHPINV(
            rram_interface=self.rram_interface,
            fault_tolerance_strategy=best_strategy
        )
        
        result = ft_hpiv.solve(G, b, **kwargs)
        
        # Learn from the outcome
        self._learn_from_outcome(result, best_strategy)
        
        return result
    
    def _learn_from_outcome(self, 
                           result: FaultToleranceResult, 
                           strategy: FaultToleranceStrategy):
        """
        Update strategy effectiveness based on outcome.
        
        Args:
            result: Result of the computation
            strategy: Strategy that was used
        """
        # Calculate a reward based on success and efficiency
        if result.success:
            # Reward successful solutions, penalize high resource usage
            reward = 1.0 / (1 + result.recovery_attempts * 0.1)  # More attempts = less reward
            reward /= (1 + result.execution_time)  # Slower = less reward
        else:
            reward = -1.0  # Penalty for failure
        
        # Update strategy weight
        self.strategy_weights[strategy] += self.learning_rate * reward
        
        # Keep weights positive
        self.strategy_weights[strategy] = max(0.1, self.strategy_weights[strategy])
        
        # Store for history
        self.performance_history.append({
            'strategy': strategy,
            'success': result.success,
            'execution_time': result.execution_time,
            'recovery_attempts': result.recovery_attempts,
            'reward': reward,
            'timestamp': time.time()
        })
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]


class FaultTolerantNeuralNetwork:
    """
    Fault-tolerant neural network using RRAM properties.
    """
    
    def __init__(self,
                 layers_config: List[Dict[str, Any]],
                 rram_interface: Optional[object] = None,
                 fault_tolerance_strategy: FaultToleranceStrategy = FaultToleranceStrategy.REDUNDANCY):
        """
        Initialize fault-tolerant neural network.
        
        Args:
            layers_config: Configuration for each layer
            rram_interface: Optional RRAM interface
            fault_tolerance_strategy: Strategy for handling faults
        """
        self.layers_config = layers_config
        self.rram_interface = rram_interface
        self.fault_tolerance_strategy = fault_tolerance_strategy
        self.fault_detector = FaultDetector()
        self.layers = []
        self._build_network()
    
    def _build_network(self):
        """Build the neural network with fault-tolerant layers."""
        for i, config in enumerate(self.layers_config):
            # Create a fault-tolerant version of the layer
            if config.get('type', 'linear') == 'rram_spiking':
                layer = RRAMSpikingLayer(
                    n_inputs=config.get('n_inputs', 10),
                    n_neurons=config.get('n_neurons', 10),
                    rram_interface=self.rram_interface
                )
            else:
                # Standard RRAM linear layer
                layer = self._create_rram_linear_layer(config)
            
            self.layers.append(layer)
    
    def _create_rram_linear_layer(self, config: Dict[str, Any]):
        """Create an RRAM linear layer with fault tolerance."""
        # For this example, we'll use a simple implementation
        # In practice, this would involve creating a layer that can handle RRAM non-idealities
        from .ml_framework_integration import TorchRRAMLinear
        try:
            import torch.nn as nn
            # Create a composite layer that includes fault tolerance
            layer = nn.Sequential(
                TorchRRAMLinear(
                    in_features=config.get('in_features', 10),
                    out_features=config.get('out_features', 10),
                    rram_interface=self.rram_interface
                ),
                nn.ReLU()  # Add non-linearity
            )
            return layer
        except ImportError:
            # Fallback if PyTorch is not available
            return None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass with fault tolerance.
        
        Args:
            input_data: Input data
            
        Returns:
            Output from the network
        """
        output = input_data
        
        for i, layer in enumerate(self.layers):
            if layer is None:
                continue
                
            # Check for faults in layer weights if interface is available
            if self.rram_interface is not None:
                try:
                    # Read current weights from RRAM
                    current_weights = self.rram_interface.read_matrix()
                    
                    # Detect faults in weights
                    fault_maps = self.fault_detector.analyze_matrix_health(current_weights)
                    
                    # Apply fault tolerance if needed
                    if any(np.any(mask) for mask in fault_maps.values()):
                        # Repair weights using redundancy or other method
                        repaired_weights = self._apply_weight_repair(current_weights, fault_maps)
                        
                        # Write repaired weights back to RRAM
                        self.rram_interface.write_matrix(repaired_weights)
                        
                except Exception:
                    # If hardware interface fails, continue with software
                    pass
            
            # Perform layer operation
            if hasattr(layer, 'forward'):
                output = layer.forward(output)
            else:
                # Fallback for numpy operations
                output = output  # Placeholder
        
        return output
    
    def _apply_weight_repair(self, 
                           weights: np.ndarray, 
                           fault_maps: Dict[FaultType, np.ndarray]) -> np.ndarray:
        """
        Apply repair to faulty weights.
        
        Args:
            weights: Weight matrix with potential faults
            fault_maps: Maps of detected faults
            
        Returns:
            Repaired weight matrix
        """
        # Combine all fault locations
        combined_faults = np.zeros_like(weights, dtype=bool)
        for fault_mask in fault_maps.values():
            combined_faults = np.logical_or(combined_faults, fault_mask)
        
        # Repair using appropriate method based on strategy
        repaired_weights = weights.copy()
        n, m = weights.shape
        
        for i in range(n):
            for j in range(m):
                if combined_faults[i, j]:
                    # Use average of neighbors for repair
                    neighbors = []
                    if i > 0: neighbors.append(weights[i-1, j])
                    if i < n-1: neighbors.append(weights[i+1, j])
                    if j > 0: neighbors.append(weights[i, j-1])
                    if j < m-1: neighbors.append(weights[i, j+1])
                    
                    if neighbors:
                        repaired_weights[i, j] = np.mean(neighbors)
                    else:
                        repaired_weights[i, j] = np.mean(weights)  # Global average
        
        return repaired_weights


def create_fault_tolerant_system(rram_interface: Optional[object] = None) -> Tuple[FaultTolerantHPINV, AdaptiveFaultTolerance]:
    """
    Create a fault-tolerant system with both basic and adaptive capabilities.
    
    Args:
        rram_interface: Optional RRAM interface
        
    Returns:
        Tuple of (basic fault-tolerant solver, adaptive fault-tolerant system)
    """
    basic_solver = FaultTolerantHPINV(rram_interface=rram_interface)
    adaptive_system = AdaptiveFaultTolerance(rram_interface=rram_interface)
    
    return basic_solver, adaptive_system


def demonstrate_fault_tolerance():
    """
    Demonstrate fault tolerance capabilities.
    """
    print("Demonstrating Fault-Tolerant Algorithms...")
    
    # Create a matrix with some stuck-at faults
    n = 8
    G = np.random.rand(n, n) * 1e-4
    G = G + 0.5 * np.eye(n)  # Diagonally dominant
    
    # Simulate some stuck-at faults
    G[0, 1] = 0  # Stuck at low conductance
    G[2, 3] = 0  # Another stuck fault
    G[5, 5] = 0  # Stuck diagonal element
    
    b = np.random.rand(n)
    
    print(f"Original matrix condition: {np.linalg.cond(G):.2e}")
    print(f"Faults in matrix: {np.sum(G == 0)} stuck elements")
    
    # Test fault-tolerant solver with redundancy strategy
    ft_solver = FaultTolerantHPINV(
        fault_tolerance_strategy=FaultToleranceStrategy.REDUNDANCY
    )
    
    result = ft_solver.solve(G, b, max_iter=20, tol=1e-6)
    
    print(f"Solution successful: {result.success}")
    print(f"Recovery attempts: {result.recovery_attempts}")
    print(f"Execution time: {result.execution_time:.4f}s")
    print(f"Residual norm: {result.error_metrics['residual_norm']:.2e}")
    
    # Validate solution
    residual = np.linalg.norm(G @ result.solution - b)
    print(f"Actual residual: {residual:.2e}")
    
    # Test adaptive system
    print("\nTesting adaptive fault tolerance...")
    adaptive_system = AdaptiveFaultTolerance()
    adaptive_result = adaptive_system.solve_adaptively(G, b, max_iter=20, tol=1e-6)
    
    print(f"Adaptive solution successful: {adaptive_result.success}")
    print(f"Strategy used: {adaptive_result.recovery_strategy.value}")
    print(f"Adaptive residual: {adaptive_result.error_metrics['residual_norm']:.2e}")

if __name__ == "__main__":
    demonstrate_fault_tolerance()