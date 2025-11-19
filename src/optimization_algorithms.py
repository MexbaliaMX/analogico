"""
Optimization Algorithms for RRAM-based Computing.

This module provides advanced optimization algorithms that take advantage
of RRAM properties, including automatic parameter tuning, machine learning-based
optimization, and adaptive algorithms based on matrix properties.
"""
import numpy as np
import warnings
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from abc import ABC, abstractmethod
import time
from enum import Enum

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV, AdaptivePrecisionHPINV
    from .neuromorphic_modules import RRAMSpikingLayer, RRAMReservoirComputer
    from .model_deployment_pipeline import RRAMModelDeployer
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class OptimizationTarget(Enum):
    """Target for optimization."""
    CONVERGENCE_SPEED = "convergence_speed"
    ACCURACY = "accuracy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    STABILITY = "stability"


class RRAMOptimizer(ABC):
    """
    Abstract base class for RRAM-specific optimizers.
    """
    
    @abstractmethod
    def optimize(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Optimize the solution for Gx = b using RRAM-specific techniques.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimized solution and metadata
        """
        pass


class ParameterTuningOptimizer(RRAMOptimizer):
    """
    Optimizer that automatically tunes HP-INV parameters based on matrix properties
    and RRAM characteristics.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 max_bits: int = 8,
                 max_iterations: int = 20):
        """
        Initialize the parameter tuning optimizer.
        
        Args:
            rram_interface: Optional RRAM interface for hardware feedback
            max_bits: Maximum quantization bits to consider
            max_iterations: Maximum number of iterations to try
        """
        self.rram_interface = rram_interface
        self.max_bits = max_bits
        self.max_iterations = max_iterations
        self.optimization_history = []
    
    def optimize(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Optimize HP-INV parameters based on matrix properties.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimized solution and parameters
        """
        # Analyze matrix properties to determine optimal parameters
        matrix_properties = self._analyze_matrix_properties(G)
        
        # Determine optimal parameters based on matrix properties and RRAM characteristics
        optimal_params = self._determine_optimal_parameters(matrix_properties)
        
        # Apply RRAM-specific adjustments if interface is available
        if self.rram_interface is not None:
            optimal_params = self._adjust_for_rram_hardware(optimal_params)
        
        # Solve using optimized parameters
        solution, iterations, info = hp_inv(G, b, **optimal_params)
        
        result = {
            'solution': solution,
            'iterations': iterations,
            'info': info,
            'optimized_params': optimal_params,
            'matrix_properties': matrix_properties
        }
        
        self.optimization_history.append(result)
        
        return result
    
    def _analyze_matrix_properties(self, G: np.ndarray) -> Dict[str, float]:
        """
        Analyze matrix properties to inform parameter selection.
        
        Args:
            G: Matrix to analyze
            
        Returns:
            Dictionary of matrix properties
        """
        properties = {}
        
        # Condition number (affects stability and required precision)
        properties['condition_number'] = float(np.linalg.cond(G))
        
        # Sparsity (affects optimal blocking and computation strategy)
        total_elements = G.size
        zero_elements = np.count_nonzero(G == 0)
        properties['sparsity'] = float(zero_elements / total_elements)
        
        # Spectral properties
        eigenvalues = np.linalg.eigvals(G)
        properties['eigenvalue_range'] = float(np.max(np.abs(eigenvalues)) / 
                                             np.min(np.abs(eigenvalues[np.nonzero(eigenvalues)])))
        properties['max_eigenvalue'] = float(np.max(np.abs(eigenvalues)))
        
        # Matrix norms
        properties['l1_norm'] = float(np.linalg.norm(G, 1))
        properties['l2_norm'] = float(np.linalg.norm(G, 2))
        properties['frobenius_norm'] = float(np.linalg.norm(G, 'fro'))
        
        # Diagonal dominance
        diagonal_elements = np.abs(np.diag(G))
        off_diagonal_row_sum = np.sum(np.abs(G), axis=1) - diagonal_elements
        diagonal_dominance = diagonal_elements / (off_diagonal_row_sum + 1e-12)  # Avoid division by zero
        properties['min_diagonal_dominance'] = float(np.min(diagonal_dominance))
        properties['avg_diagonal_dominance'] = float(np.mean(diagonal_dominance))
        
        return properties
    
    def _determine_optimal_parameters(self, properties: Dict[str, float]) -> Dict[str, Any]:
        """
        Determine optimal HP-INV parameters based on matrix properties.
        
        Args:
            properties: Matrix properties
            
        Returns:
            Dictionary of optimal parameters
        """
        params = {}
        
        # Determine quantization bits based on condition number
        cond_num = properties['condition_number']
        if cond_num < 1e3:
            params['bits'] = 3
        elif cond_num < 1e6:
            params['bits'] = 4
        elif cond_num < 1e9:
            params['bits'] = 5
        else:
            params['bits'] = 6  # Maximum practical precision
        
        # Limit to specified maximum
        params['bits'] = min(params['bits'], self.max_bits)
        
        # Determine relaxation factor based on diagonal dominance
        diag_dominance = properties['avg_diagonal_dominance']
        if diag_dominance > 5.0:
            params['relaxation_factor'] = 1.0  # No relaxation needed for strongly diagonally dominant
        elif diag_dominance > 1.0:
            params['relaxation_factor'] = 0.9   # Slight under-relaxation
        else:
            params['relaxation_factor'] = 0.7   # More conservative relaxation
        
        # Determine noise level based on condition number
        params['lp_noise_std'] = min(0.01 * (cond_num / 1e3), 0.1)  # Scale noise with condition
        
        # Maximum iterations based on problem difficulty
        if cond_num < 1e4:
            params['max_iter'] = 10
        elif cond_num < 1e8:
            params['max_iter'] = 15
        else:
            params['max_iter'] = 20
        
        params['max_iter'] = min(params['max_iter'], self.max_iterations)
        
        # Tolerance based on desired accuracy
        params['tol'] = 1e-6  # Standard tolerance
        
        return params
    
    def _adjust_for_rram_hardware(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters for specific RRAM hardware characteristics.
        
        Args:
            params: Original parameters
            
        Returns:
            Adjusted parameters
        """
        adjusted_params = params.copy()
        
        # If we have access to the RRAM interface, get its characteristics
        if hasattr(self.rram_interface, 'variability'):
            # Adjust noise parameters based on hardware variability
            adjusted_params['lp_noise_std'] = max(
                adjusted_params['lp_noise_std'], 
                self.rram_interface.variability
            )
        
        if hasattr(self.rram_interface, 'stuck_fault_prob'):
            # Adjust if there are significant stuck-at faults
            if self.rram_interface.stuck_fault_prob > 0.02:  # More than 2% faults
                adjusted_params['max_iter'] = min(adjusted_params['max_iter'], 10)
        
        return adjusted_params


class MachineLearningOptimizer(RRAMOptimizer):
    """
    Machine learning-based optimizer that learns optimal parameters from experience.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 learning_rate: float = 0.01,
                 use_gpu: bool = False):
        """
        Initialize the ML-based optimizer.
        
        Args:
            rram_interface: Optional RRAM interface for hardware feedback
            learning_rate: Learning rate for parameter updates
            use_gpu: Whether to use GPU acceleration if available
        """
        self.rram_interface = rram_interface
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.experience_buffer = []
        self.parameter_weights = {
            'bits': 1.0,
            'relaxation_factor': 1.0,
            'lp_noise_std': 1.0,
            'max_iter': 1.0
        }
        self.performance_history = []
    
    def optimize(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Optimize using ML-based parameter selection.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimized solution and parameters
        """
        # Extract features from the matrix
        features = self._extract_matrix_features(G, b)
        
        # Predict optimal parameters based on features and learned weights
        predicted_params = self._predict_parameters(features)
        
        # Apply any constraints
        constrained_params = self._apply_parameter_constraints(predicted_params)
        
        # Solve with predicted parameters
        solution, iterations, info = hp_inv(G, b, **constrained_params)
        
        # Calculate performance metrics
        performance = self._evaluate_performance(solution, G, b, iterations, info)
        
        # Store experience for learning
        self.experience_buffer.append({
            'features': features,
            'parameters': constrained_params,
            'performance': performance,
            'timestamp': time.time()
        })
        
        # Update learned parameter weights based on performance
        self._update_parameter_weights(constrained_params, performance)
        
        # Limit experience buffer size
        if len(self.experience_buffer) > 1000:
            self.experience_buffer.pop(0)
        
        result = {
            'solution': solution,
            'iterations': iterations,
            'info': info,
            'predicted_params': constrained_params,
            'features': features,
            'performance': performance
        }
        
        self.performance_history.append(performance)
        
        return result
    
    def _extract_matrix_features(self, G: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Extract features from the matrix system for ML model.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Basic matrix properties
        features.append(np.log10(max(np.linalg.cond(G), 1e-12)))  # Log condition number
        features.append(np.linalg.norm(G, 'fro'))  # Frobenius norm
        features.append(np.linalg.norm(b))  # Norm of b
        features.append(G.shape[0])  # Matrix size
        
        # Spectral properties
        eigenvals = np.linalg.eigvals(G)
        features.append(np.log10(max(np.max(np.abs(eigenvals)), 1e-12)))  # Max eigenvalue (log)
        features.append(np.log10(max(np.min(np.abs(eigenvals[eigenvals!=0])), 1e-12)))  # Min eigenvalue (log)
        
        # Diagonal properties
        diag_elements = np.diag(G)
        features.append(np.mean(np.abs(diag_elements)))  # Avg diagonal abs value
        features.append(np.std(diag_elements))  # Std of diagonal values
        
        # Sparsity
        features.append(np.count_nonzero(G) / G.size)  # Sparsity ratio
        
        # Off-diagonal dominance
        off_diag_sum = np.sum(np.abs(G - np.diag(diag_elements)), axis=1)
        diag_abs = np.abs(diag_elements)
        dominance_ratio = np.mean(diag_abs / (off_diag_sum + 1e-12))  # Avoid division by zero
        features.append(dominance_ratio)
        
        return np.array(features)
    
    def _predict_parameters(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict optimal parameters based on features and learned weights.
        
        Args:
            features: Matrix system features
            
        Returns:
            Predicted parameters dictionary
        """
        # This is a simplified linear model - in practice, you could use more complex models
        base_params = {
            'bits': 4.0,
            'relaxation_factor': 1.0,
            'lp_noise_std': 0.01,
            'max_iter': 10.0
        }
        
        # Apply learned adjustments based on features
        adjusted_params = base_params.copy()
        
        # Adjust bits based on condition number (first feature)
        cond_factor = self.parameter_weights['bits'] * features[0] * 0.1
        adjusted_params['bits'] = max(2, min(8, 4 + cond_factor))
        
        # Adjust relaxation based on diagonal dominance (last feature)
        dom_factor = self.parameter_weights['relaxation_factor'] * features[-1] * 0.2
        adjusted_params['relaxation_factor'] = max(0.1, min(1.5, 1.0 + dom_factor))
        
        # Adjust noise based on matrix size and condition
        size_cond_factor = self.parameter_weights['lp_noise_std'] * (features[3] * features[0]) * 0.001
        adjusted_params['lp_noise_std'] = max(1e-4, min(0.1, 0.01 + size_cond_factor))
        
        # Adjust iterations based on condition number
        iter_factor = self.parameter_weights['max_iter'] * features[0] * 0.5
        adjusted_params['max_iter'] = max(5, min(50, 10 + iter_factor))
        
        return {k: float(v) for k, v in adjusted_params.items()}
    
    def _apply_parameter_constraints(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Apply practical constraints to predicted parameters.
        
        Args:
            params: Predicted parameters
            
        Returns:
            Constrained parameters
        """
        constrained = params.copy()
        
        # Quantization bits must be integers between 1 and 8
        constrained['bits'] = int(max(1, min(8, round(constrained['bits']))))
        
        # Relaxation factor should be in reasonable range
        constrained['relaxation_factor'] = np.clip(constrained['relaxation_factor'], 0.1, 2.0)
        
        # Noise level should be positive and reasonable
        constrained['lp_noise_std'] = np.clip(constrained['lp_noise_std'], 1e-6, 1.0)
        
        # Max iterations should be a reasonable integer
        constrained['max_iter'] = int(max(1, min(100, round(constrained['max_iter']))))
        
        return constrained
    
    def _evaluate_performance(self, 
                             solution: np.ndarray, 
                             G: np.ndarray, 
                             b: np.ndarray, 
                             iterations: int, 
                             info: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the performance of a solution.
        
        Args:
            solution: Computed solution vector
            G: Coefficient matrix
            b: Right-hand side vector
            iterations: Number of iterations taken
            info: Additional solution info
            
        Returns:
            Performance metrics dictionary
        """
        # Calculate residual
        residual = G @ solution - b
        residual_norm = np.linalg.norm(residual)
        
        # Calculate relative error if true solution is known (for synthetic problems)
        # In practice, we'd only have the residual
        performance = {
            'residual_norm': float(residual_norm),
            'iterations': iterations,
            'converged': info.get('converged', False),
            'final_tolerance': float(info.get('final_residual', residual_norm)),
            'time_to_solution': info.get('time_taken', 0),
            'efficiency_score': float(iterations / (max(residual_norm, 1e-12) + 1))  # Inverse of effort per accuracy
        }
        
        return performance
    
    def _update_parameter_weights(self, 
                                 params: Dict[str, float], 
                                 performance: Dict[str, float]):
        """
        Update learned parameter weights based on performance feedback.
        
        Args:
            params: Parameters used
            performance: Performance achieved
        """
        # Simple gradient-based update (in practice, more sophisticated methods could be used)
        # Reward parameters that led to good performance (low residual, few iterations)
        reward = 1.0 / (performance['residual_norm'] * performance['iterations'] + 1e-6)
        
        # Update weights based on the reward and parameter values
        for param_name in self.parameter_weights:
            if param_name in params:
                # Adjust weights proportionally to parameter values and performance
                adjustment = self.learning_rate * reward * params[param_name]
                self.parameter_weights[param_name] += adjustment


class AdaptiveMatrixPropertyOptimizer(RRAMOptimizer):
    """
    Optimizer that adapts its strategy based on matrix properties.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 optimization_target: OptimizationTarget = OptimizationTarget.CONVERGENCE_SPEED):
        """
        Initialize the matrix property-based optimizer.
        
        Args:
            rram_interface: Optional RRAM interface
            optimization_target: What to optimize for
        """
        self.rram_interface = rram_interface
        self.optimization_target = optimization_target
    
    def optimize(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Optimize based on matrix properties.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimized solution
        """
        # Analyze matrix for optimal approach
        matrix_type = self._classify_matrix_type(G)
        
        if matrix_type == 'diagonally_dominant':
            return self._solve_diagonally_dominant(G, b, **kwargs)
        elif matrix_type == 'sparse':
            return self._solve_sparse(G, b, **kwargs)
        elif matrix_type == 'ill_conditioned':
            return self._solve_ill_conditioned(G, b, **kwargs)
        elif matrix_type == 'symmetric_positive_definite':
            return self._solve_symmetric_positive_definite(G, b, **kwargs)
        else:
            # Default to general approach
            return self._solve_general(G, b, **kwargs)
    
    def _classify_matrix_type(self, G: np.ndarray) -> str:
        """
        Classify the matrix type to select the optimal approach.
        
        Args:
            G: Matrix to classify
            
        Returns:
            String representing the matrix type
        """
        # Check diagonal dominance
        diagonal_elements = np.abs(np.diag(G))
        off_diagonal_row_sum = np.sum(np.abs(G), axis=1) - diagonal_elements
        diagonal_dominance_ratio = diagonal_elements / (off_diagonal_row_sum + 1e-12)
        
        is_diagonally_dominant = np.all(diagonal_dominance_ratio > 1.0)
        
        # Check sparsity
        sparsity = np.count_nonzero(G) / G.size
        is_sparse = sparsity < 0.1  # Less than 10% non-zero
        
        # Check condition number
        condition_number = np.linalg.cond(G)
        is_ill_conditioned = condition_number > 1e12
        
        # Check if symmetric and positive definite
        is_symmetric = np.allclose(G, G.T)
        is_positive_definite = False
        if is_symmetric:
            try:
                # Check if all eigenvalues are positive
                eigenvals = np.linalg.eigvals(G)
                is_positive_definite = np.all(eigenvals > 0)
            except:
                is_positive_definite = False
        
        # Classify based on properties
        if is_diagonally_dominant:
            return 'diagonally_dominant'
        elif is_sparse:
            return 'sparse'
        elif is_ill_conditioned:
            return 'ill_conditioned'
        elif is_symmetric and is_positive_definite:
            return 'symmetric_positive_definite'
        else:
            return 'general'
    
    def _solve_diagonally_dominant(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Solve diagonally dominant system."""
        # For diagonally dominant matrices, standard iterative methods work well
        # Use low precision and relaxation factor close to 1.0
        params = {
            'bits': 3,
            'relaxation_factor': 0.95,
            'lp_noise_std': 0.005,
            'max_iter': 15,
            'tol': 1e-6
        }
        params.update(kwargs)
        
        solution, iterations, info = hp_inv(G, b, **params)
        
        return {
            'solution': solution,
            'iterations': iterations,
            'info': info,
            'approach': 'diagonally_dominant',
            'params_used': params
        }
    
    def _solve_sparse(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Solve sparse system."""
        # For sparse matrices, consider special techniques
        # This is a simplified approach - in practice, sparse-specific methods would be better
        params = {
            'bits': 4,
            'relaxation_factor': 1.0,
            'lp_noise_std': 0.01,
            'max_iter': 20,
            'tol': 1e-6
        }
        params.update(kwargs)
        
        solution, iterations, info = hp_inv(G, b, **params)
        
        return {
            'solution': solution,
            'iterations': iterations,
            'info': info,
            'approach': 'sparse',
            'params_used': params
        }
    
    def _solve_ill_conditioned(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Solve ill-conditioned system."""
        # For ill-conditioned matrices, use higher precision and more iterations
        params = {
            'bits': 6,
            'relaxation_factor': 0.7,  # More conservative relaxation
            'lp_noise_std': 0.001,     # Lower noise
            'max_iter': 25,
            'tol': 1e-8               # Stricter tolerance
        }
        params.update(kwargs)
        
        solution, iterations, info = adaptive_hp_inv(
            G, b, 
            initial_bits=params['bits'],
            max_bits=8,
            **{k: v for k, v in params.items() if k not in ['bits']}
        )
        
        return {
            'solution': solution,
            'iterations': iterations,
            'info': info,
            'approach': 'ill_conditioned',
            'params_used': params
        }
    
    def _solve_symmetric_positive_definite(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Solve symmetric positive definite system."""
        # For SPD matrices, Cholesky decomposition is theoretically optimal,
        # but we'll optimize HP-INV parameters for this case
        params = {
            'bits': 4,
            'relaxation_factor': 1.1,   # Slightly over-relaxation can work well for SPD
            'lp_noise_std': 0.005,
            'max_iter': 12,
            'tol': 1e-7
        }
        params.update(kwargs)
        
        solution, iterations, info = hp_inv(G, b, **params)
        
        return {
            'solution': solution,
            'iterations': iterations,
            'info': info,
            'approach': 'symmetric_positive_definite',
            'params_used': params
        }
    
    def _solve_general(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Solve general system."""
        # For general matrices, use balanced parameters
        params = {
            'bits': 4,
            'relaxation_factor': 0.9,
            'lp_noise_std': 0.01,
            'max_iter': 20,
            'tol': 1e-6
        }
        params.update(kwargs)
        
        solution, iterations, info = hp_inv(G, b, **params)
        
        return {
            'solution': solution,
            'iterations': iterations,
            'info': info,
            'approach': 'general',
            'params_used': params
        }


class MultiObjectiveOptimizer(RRAMOptimizer):
    """
    Optimizer that balances multiple objectives like speed, accuracy, and stability.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 objectives_weights: Optional[Dict[OptimizationTarget, float]] = None):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            rram_interface: Optional RRAM interface
            objectives_weights: Weights for different optimization targets
        """
        self.rram_interface = rram_interface
        self.objectives_weights = objectives_weights or {
            OptimizationTarget.CONVERGENCE_SPEED: 0.4,
            OptimizationTarget.ACCURACY: 0.4,
            OptimizationTarget.STABILITY: 0.2
        }
    
    def optimize(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Optimize balancing multiple objectives.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimized solution
        """
        # Define a range of parameter combinations to try
        param_combinations = self._generate_parameter_combinations(G)
        
        best_score = float('-inf')
        best_result = None
        best_params = None
        
        # Evaluate each parameter combination
        for params in param_combinations:
            try:
                solution, iterations, info = hp_inv(G, b, **params)
                
                # Calculate multi-objective score
                score = self._calculate_multi_objective_score(
                    solution, G, b, iterations, info, params
                )
                
                if score > best_score:
                    best_score = score
                    best_result = (solution, iterations, info)
                    best_params = params
                    
            except Exception:
                # Skip invalid parameter combinations
                continue
        
        if best_result is None:
            # Fallback to default parameters if no valid combination found
            default_params = {
                'bits': 4,
                'relaxation_factor': 0.9,
                'lp_noise_std': 0.01,
                'max_iter': 15,
                'tol': 1e-6
            }
            solution, iterations, info = hp_inv(G, b, **default_params)
            best_result = (solution, iterations, info)
            best_params = default_params
        
        solution, iterations, info = best_result
        
        return {
            'solution': solution,
            'iterations': iterations,
            'info': info,
            'optimal_params': best_params,
            'multi_objective_score': best_score,
            'param_search_space': param_combinations
        }
    
    def _generate_parameter_combinations(self, G: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate reasonable parameter combinations to try.
        
        Args:
            G: Matrix to inform parameter selection
            
        Returns:
            List of parameter dictionaries
        """
        # Analyze matrix to inform parameter ranges
        cond_num = np.linalg.cond(G)
        
        if cond_num < 1e4:
            bits_range = [3, 4, 5]
            max_iter_range = [8, 12, 15]
        elif cond_num < 1e8:
            bits_range = [4, 5, 6]
            max_iter_range = [12, 15, 20]
        else:
            bits_range = [5, 6, 7]
            max_iter_range = [15, 20, 25]
        
        relaxation_range = [0.7, 0.85, 1.0, 1.15]
        noise_range = [0.001, 0.005, 0.01, 0.02]
        
        combinations = []
        for bits in bits_range:
            for max_iter in max_iter_range:
                for relaxation in relaxation_range:
                    for noise in noise_range:
                        combinations.append({
                            'bits': bits,
                            'relaxation_factor': relaxation,
                            'lp_noise_std': noise,
                            'max_iter': max_iter,
                            'tol': 1e-6
                        })
        
        return combinations
    
    def _calculate_multi_objective_score(self,
                                       solution: np.ndarray,
                                       G: np.ndarray,
                                       b: np.ndarray,
                                       iterations: int,
                                       info: Dict[str, Any],
                                       params: Dict[str, Any]) -> float:
        """
        Calculate a multi-objective score balancing speed, accuracy, and stability.
        
        Args:
            solution: Computed solution
            G: Coefficient matrix
            b: Right-hand side vector
            iterations: Number of iterations
            info: Additional solution info
            params: Parameters used
            
        Returns:
            Multi-objective score (higher is better)
        """
        # Calculate individual objective values (normalize to 0-1 range)
        
        # Accuracy: negative log of residual (higher is better)
        residual = G @ solution - b
        residual_norm = np.linalg.norm(residual)
        accuracy_score = -np.log10(max(residual_norm, 1e-15))  # Higher is better
        
        # Speed: inverse of iterations (higher is better)
        speed_score = 1.0 / max(iterations, 1)  # Higher is better
        
        # Stability: based on relaxation factor and convergence behavior
        stability_score = 1.0 - abs(params['relaxation_factor'] - 1.0)  # Closer to 1.0 is more stable
        if not info.get('converged', False):
            stability_score *= 0.5  # Penalize non-convergence
        
        # Normalize scores to reasonable ranges
        accuracy_score = max(0, min(10, accuracy_score + 10)) / 10  # Normalize to 0-1
        speed_score = max(0, min(1, speed_score * 10))  # Normalize to 0-1
        stability_score = max(0, stability_score)  # Already in 0-1 range
        
        # Calculate weighted score
        total_weight = sum(self.objectives_weights.values())
        score = (
            self.objectives_weights[OptimizationTarget.ACCURACY] * accuracy_score +
            self.objectives_weights[OptimizationTarget.CONVERGENCE_SPEED] * speed_score +
            self.objectives_weights[OptimizationTarget.STABILITY] * stability_score
        ) / total_weight
        
        return score


class RRAMAwareOptimizer:
    """
    Main interface for RRAM-aware optimization that combines all approaches.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 optimization_target: OptimizationTarget = OptimizationTarget.CONVERGENCE_SPEED):
        """
        Initialize the RRAM-aware optimizer.
        
        Args:
            rram_interface: Optional RRAM interface
            optimization_target: Primary optimization target
        """
        self.rram_interface = rram_interface
        self.optimization_target = optimization_target
        
        # Initialize all optimizer types
        self.parameter_tuner = ParameterTuningOptimizer(rram_interface)
        self.ml_optimizer = MachineLearningOptimizer(rram_interface)
        self.matrix_property_optimizer = AdaptiveMatrixPropertyOptimizer(
            rram_interface, optimization_target
        )
        self.multi_objective_optimizer = MultiObjectiveOptimizer(rram_interface)
    
    def optimize(self, 
                G: np.ndarray, 
                b: np.ndarray, 
                method: str = 'auto',
                **kwargs) -> Dict[str, Any]:
        """
        Optimize using the specified method or automatically selected method.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            method: Optimization method ('auto', 'parameter_tuning', 'ml', 'matrix_property', 'multi_objective')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimized solution and metadata
        """
        if method == 'auto':
            # Automatically select the best method based on matrix properties
            method = self._select_optimal_method(G)
        
        start_time = time.time()
        
        if method == 'parameter_tuning':
            result = self.parameter_tuner.optimize(G, b, **kwargs)
        elif method == 'ml':
            result = self.ml_optimizer.optimize(G, b, **kwargs)
        elif method == 'matrix_property':
            result = self.matrix_property_optimizer.optimize(G, b, **kwargs)
        elif method == 'multi_objective':
            result = self.multi_objective_optimizer.optimize(G, b, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        result['optimization_method'] = method
        result['optimization_time'] = time.time() - start_time
        
        return result
    
    def _select_optimal_method(self, G: np.ndarray) -> str:
        """
        Automatically select the optimal optimization method based on matrix properties.
        
        Args:
            G: Matrix to analyze
            
        Returns:
            Selected method name
        """
        # Analyze matrix properties to decide on method
        cond_num = np.linalg.cond(G)
        sparsity = np.count_nonzero(G) / G.size
        is_symmetric = np.allclose(G, G.T)
        
        if cond_num > 1e12:
            # Very ill-conditioned - use multi-objective to balance stability and accuracy
            return 'multi_objective'
        elif sparsity < 0.1:
            # Very sparse - matrix property based might be good
            return 'matrix_property'
        elif is_symmetric:
            # Symmetric matrix - parameter tuning might work well
            return 'parameter_tuning'
        else:
            # General case - ML-based optimizer learns from experience
            return 'ml'
    
    def benchmark_methods(self, G: np.ndarray, b: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark all optimization methods on the given system.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            
        Returns:
            Dictionary of results for each method
        """
        methods = ['parameter_tuning', 'ml', 'matrix_property', 'multi_objective']
        results = {}
        
        for method in methods:
            try:
                results[method] = self.optimize(G, b, method=method)
            except Exception as e:
                results[method] = {
                    'error': str(e),
                    'success': False
                }
        
        return results


def create_rram_aware_optimizer( 
    rram_interface: Optional[object] = None,
    optimization_target: OptimizationTarget = OptimizationTarget.CONVERGENCE_SPEED
) -> RRAMAwareOptimizer:
    """
    Factory function to create an RRAM-aware optimizer.
    
    Args:
        rram_interface: Optional RRAM interface
        optimization_target: Primary optimization target
        
    Returns:
        RRAMAwareOptimizer instance
    """
    return RRAMAwareOptimizer(rram_interface, optimization_target)


def optimize_rram_system(G: np.ndarray, 
                        b: np.ndarray, 
                        rram_interface: Optional[object] = None,
                        target: OptimizationTarget = OptimizationTarget.CONVERGENCE_SPEED) -> Dict[str, Any]:
    """
    Convenience function to optimize a system using RRAM-aware optimization.
    
    Args:
        G: Coefficient matrix
        b: Right-hand side vector
        rram_interface: Optional RRAM interface
        target: Optimization target
        
    Returns:
        Optimization result
    """
    optimizer = create_rram_aware_optimizer(rram_interface, target)
    return optimizer.optimize(G, b)