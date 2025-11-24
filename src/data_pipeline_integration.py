"""
Data Pipeline Integration for RRAM-based Computing Systems.

This module provides tools for integrating RRAM computations into
data science and machine learning pipelines, including data loaders,
transformers, and pipeline components.
"""
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union, Callable, Iterator
import time
import os
from pathlib import Path
import pickle
import json
from datetime import datetime
from functools import wraps

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV
    from .neuromorphic_modules import RRAMNeuromorphicNetwork, RRAMSpikingLayer
    from .optimization_algorithms import RRAMAwareOptimizer
    from .benchmarking_suite import RRAMBenchmarkSuite
    from .visualization_tools import RRAMVisualizer
    from .fault_tolerant_algorithms import FaultTolerantHPINV, AdaptiveFaultTolerance
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class RRAMDataLoader:
    """
    Data loader for RRAM-specific formats and operations.
    """
    
    def __init__(self, 
                 batch_size: int = 32,
                 shuffle: bool = True,
                 rram_interface: Optional[object] = None):
        """
        Initialize the RRAM data loader.
        
        Args:
            batch_size: Size of data batches
            shuffle: Whether to shuffle data
            rram_interface: Optional RRAM interface for hardware operations
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rram_interface = rram_interface
        self.data_cache = {}
    
    def load_matrix_data(self, 
                        file_path: str, 
                        matrix_type: str = 'conductance') -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Load matrix data from various formats for RRAM processing.
        
        Args:
            file_path: Path to the data file
            matrix_type: Type of matrix ('conductance', 'resistance', 'general')
            
        Yields:
            Tuples of (matrix, target) for training or processing
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.npy':
            # Single matrix file
            matrix = np.load(file_path)
            yield matrix, np.zeros(matrix.shape[0])  # No target for pure matrix
            
        elif file_path.suffix == '.npz':
            # Multiple matrices in compressed format
            with np.load(file_path) as data:
                keys = [key for key in data.keys() if key.startswith('matrix')]
                for key in keys:
                    matrix = data[key]
                    yield matrix, np.zeros(matrix.shape[0])
                    
        elif file_path.suffix == '.csv':
            # CSV format - assume it contains a single matrix
            df = pd.read_csv(file_path)
            matrix = df.values.astype(np.float32)
            yield matrix, np.zeros(matrix.shape[0])
            
        elif file_path.suffix == '.json':
            # JSON format with multiple matrices
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    if 'matrix' in item:
                        matrix = np.array(item['matrix'])
                        target = np.array(item.get('target', np.zeros(matrix.shape[0])))
                        yield matrix, target
            elif isinstance(data, dict) and 'matrices' in data:
                for matrix_data in data['matrices']:
                    matrix = np.array(matrix_data['matrix'])
                    target = np.array(matrix_data.get('target', np.zeros(matrix.shape[0])))
                    yield matrix, target
    
    def load_linear_systems(self, 
                          file_path: str, 
                          format_type: str = 'standard') -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Load linear systems (G, b) pairs from file.
        
        Args:
            file_path: Path to the data file
            format_type: Format of the data ('standard', 'numpy', 'json')
            
        Yields:
            Tuples of (G matrix, b vector) for solving Gx=b
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.npz':
            with np.load(file_path) as data:
                if 'G_matrices' in data and 'b_vectors' in data:
                    G_matrices = data['G_matrices']
                    b_vectors = data['b_vectors']
                    
                    for G, b in zip(G_matrices, b_vectors):
                        yield G, b
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    if 'G' in item and 'b' in item:
                        G = np.array(item['G'])
                        b = np.array(item['b'])
                        yield G, b
        else:
            # Default to standard numpy format
            matrices = np.load(file_path)
            # Assume alternating G, b pairs
            for i in range(0, len(matrices), 2):
                if i + 1 < len(matrices):
                    yield matrices[i], matrices[i + 1]
    
    def create_matrix_dataset(self, 
                            size: int, 
                            num_samples: int,
                            condition_range: Tuple[float, float] = (1, 1e6),
                            sparsity_range: Tuple[float, float] = (0.0, 0.1)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a synthetic dataset of matrices for RRAM training/evaluation.
        
        Args:
            size: Size of each square matrix
            num_samples: Number of matrices to generate
            condition_range: Range of condition numbers
            sparsity_range: Range of sparsity values
            
        Returns:
            Tuple of (matrices array, targets array)
        """
        matrices = []
        targets = []
        
        for _ in range(num_samples):
            # Generate matrix with specified properties
            matrix = self._generate_conditioned_matrix(size, condition_range, sparsity_range)
            
            # Create corresponding target vector (could be solution to Ax=b)
            x_true = np.random.randn(size)
            b = matrix @ x_true
            
            matrices.append(matrix)
            targets.append(b)
        
        return np.array(matrices), np.array(targets)
    
    def _generate_conditioned_matrix(self, 
                                   size: int,
                                   condition_range: Tuple[float, float],
                                   sparsity_range: Tuple[float, float]) -> np.ndarray:
        """
        Generate a matrix with specific condition number and sparsity.
        
        Args:
            size: Size of the square matrix
            condition_range: Min and max condition number
            sparsity_range: Min and max sparsity
            
        Returns:
            Generated matrix
        """
        min_cond, max_cond = condition_range
        min_sparsity, max_sparsity = sparsity_range
        
        # Generate random matrix
        matrix = np.random.randn(size, size)
        
        # Apply sparsity
        sparsity = np.random.uniform(min_sparsity, max_sparsity)
        mask = np.random.random((size, size)) < sparsity
        matrix[mask] = 0
        
        # Ensure it's not completely sparse
        if np.count_nonzero(matrix) < size:
            # Make sure diagonal is non-zero to ensure some structure
            np.fill_diagonal(matrix, np.random.randn(size) + 1)
        
        # Adjust condition number through singular value manipulation
        try:
            U, s, Vt = np.linalg.svd(matrix)
            target_cond = np.random.uniform(min_cond, max_cond)
            
            # Adjust singular values to achieve target condition number
            s_max = np.max(s)
            s_min = s_max / target_cond
            s = np.clip(s, s_min, s_max)
            
            # Reconstruct matrix
            matrix = U @ np.diag(s) @ Vt
        except np.linalg.LinAlgError:
            # If SVD fails, return random matrix with diagonal dominance
            matrix = matrix + size * np.eye(size)
        
        return matrix


class RRAMTransformer:
    """
    Transformer for preprocessing data for RRAM operations.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 normalize: bool = True,
                 scale_to_range: Optional[Tuple[float, float]] = None):
        """
        Initialize the RRAM transformer.
        
        Args:
            rram_interface: Optional RRAM interface for hardware-specific transforms
            normalize: Whether to normalize data
            scale_to_range: Range to scale values to (e.g., conductance range)
        """
        self.rram_interface = rram_interface
        self.normalize = normalize
        self.scale_to_range = scale_to_range
        self.fitted = False
        self.stats = {}
    
    def fit(self, X: np.ndarray) -> 'RRAMTransformer':
        """
        Fit the transformer to data.
        
        Args:
            X: Input data array
            
        Returns:
            Fitted transformer
        """
        if self.normalize:
            self.stats['mean'] = np.mean(X, axis=0)
            self.stats['std'] = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        
        if self.scale_to_range:
            self.stats['data_min'] = np.min(X)
            self.stats['data_max'] = np.max(X)
            self.stats['scale_min'], self.stats['scale_max'] = self.scale_to_range
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data for RRAM operations.
        
        Args:
            X: Input data array
            
        Returns:
            Transformed data array
        """
        if not self.fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        X_transformed = X.copy()
        
        # Normalize if requested
        if self.normalize and 'mean' in self.stats:
            X_transformed = (X_transformed - self.stats['mean']) / self.stats['std']
        
        # Scale to range if requested
        if self.scale_to_range and all(key in self.stats for key in ['data_min', 'data_max']):
            # Min-max scaling to target range
            data_range = self.stats['data_max'] - self.stats['data_min']
            target_range = self.stats['scale_max'] - self.stats['scale_min']
            
            X_transformed = self.stats['scale_min'] + (X_transformed - self.stats['data_min']) * (target_range / data_range)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Input data array
            
        Returns:
            Transformed data array
        """
        return self.fit(X).transform(X)
    
    def transform_for_rram(self, 
                          matrix: np.ndarray, 
                          target_range: Tuple[float, float] = (1e-6, 1e-3)) -> np.ndarray:
        """
        Transform matrix specifically for RRAM conductance values.
        
        Args:
            matrix: Input matrix
            target_range: Target range for conductance values
            
        Returns:
            Matrix transformed to RRAM-compatible values
        """
        # Scale matrix values to RRAM conductance range
        min_val, max_val = target_range
        matrix_min, matrix_max = np.min(matrix), np.max(matrix)
        
        if matrix_min == matrix_max:
            # If all values are the same, use midpoint of target range
            return np.full_like(matrix, (min_val + max_val) / 2)
        
        # Normalize to [0, 1] then scale to target range
        normalized = (matrix - matrix_min) / (matrix_max - matrix_min)
        scaled = min_val + normalized * (max_val - min_val)
        
        return scaled


class RRAMPipelineComponent:
    """
    Base class for RRAM pipeline components.
    """
    
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.fitted = False
    
    def fit(self, X, y=None, **kwargs):
        """Fit the component to data."""
        self.fitted = True
        return self
    
    def transform(self, X, **kwargs):
        """Transform the data."""
        return X
    
    def fit_transform(self, X, y=None, **kwargs):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X, **kwargs)
    
    def __call__(self, X, **kwargs):
        """Make the component callable."""
        return self.transform(X, **kwargs)


class RRAMLinearSystemSolver(RRAMPipelineComponent):
    """
    Pipeline component for solving linear systems using RRAM.
    """
    
    def __init__(self, 
                 solver_type: str = 'hp_inv',
                 rram_interface: Optional[object] = None,
                 fault_tolerance: bool = True,
                 **solver_kwargs):
        """
        Initialize the RRAM linear system solver.
        
        Args:
            solver_type: Type of solver ('hp_inv', 'block_hp_inv', 'adaptive_hp_inv')
            rram_interface: Optional RRAM interface
            fault_tolerance: Whether to use fault-tolerant algorithms
            **solver_kwargs: Additional arguments for the solver
        """
        super().__init__("RRAMLinearSystemSolver")
        self.solver_type = solver_type
        self.rram_interface = rram_interface
        self.fault_tolerance = fault_tolerance
        self.solver_kwargs = solver_kwargs
        
        # Initialize solver based on type
        if fault_tolerance:
            self.solver = FaultTolerantHPINV(rram_interface=rram_interface)
        else:
            self.solver = None  # Will use standard solvers
    
    def solve_system(self, G: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve the linear system Gx = b.
        
        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            
        Returns:
            Tuple of (solution x, info dictionary)
        """
        if self.fault_tolerance:
            result = self.solver.solve(G, b, **self.solver_kwargs)
            return result.solution, {
                'success': result.success,
                'recovery_attempts': result.recovery_attempts,
                'execution_time': result.execution_time,
                'error_metrics': result.error_metrics
            }
        else:
            # Use standard solver
            if self.solver_type == 'block_hp_inv':
                x, iterations, info = block_hp_inv(G, b, **self.solver_kwargs)
            elif self.solver_type == 'adaptive_hp_inv':
                x, iterations, info = adaptive_hp_inv(G, b, **self.solver_kwargs)
            else:  # Default to hp_inv
                x, iterations, info = hp_inv(G, b, **self.solver_kwargs)
            
            return x, {'iterations': iterations, 'info': info}
    
    def transform(self, data, **kwargs):
        """
        Transform method that expects (G, b) pairs.
        
        Args:
            data: Tuple of (G, b) or list of (G, b) pairs
            
        Returns:
            Solutions for each system
        """
        if isinstance(data, tuple) and len(data) == 2:
            # Single (G, b) pair
            G, b = data
            solution, info = self.solve_system(G, b)
            return solution, info
        elif isinstance(data, list):
            # List of (G, b) pairs
            results = []
            for G, b in data:
                solution, info = self.solve_system(G, b)
                results.append((solution, info))
            return results
        else:
            raise ValueError(f"Input must be (G, b) tuple or list of (G, b) pairs, got {type(data)}")


class RRAMNeuralLayer(RRAMPipelineComponent):
    """
    Pipeline component for RRAM-based neural network layer.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 layer_type: str = 'spiking',
                 rram_interface: Optional[object] = None,
                 **layer_kwargs):
        """
        Initialize RRAM neural layer.
        
        Args:
            input_size: Size of input
            output_size: Size of output
            layer_type: Type of layer ('spiking', 'reservoir', 'linear')
            rram_interface: Optional RRAM interface
            **layer_kwargs: Additional arguments for the layer
        """
        super().__init__("RRAMNeuralLayer")
        self.input_size = input_size
        self.output_size = output_size
        self.layer_type = layer_type
        self.rram_interface = rram_interface
        self.layer_kwargs = layer_kwargs
        
        # Initialize the appropriate layer
        if layer_type == 'spiking':
            self.layer = RRAMSpikingLayer(
                n_inputs=input_size,
                n_neurons=output_size,
                rram_interface=rram_interface,
                **layer_kwargs
            )
        elif layer_type == 'reservoir':
            from .neuromorphic_modules import RRAMReservoirComputer
            self.layer = RRAMReservoirComputer(
                input_size=input_size,
                reservoir_size=max(100, input_size * 3),  # Reservoir typically larger
                output_size=output_size,
                rram_interface=rram_interface,
                **layer_kwargs
            )
        else:  # Default to linear
            from .ml_framework_integration import TorchRRAMLinear
            try:
                import torch.nn as nn
                self.layer = nn.Sequential(
                    TorchRRAMLinear(
                        in_features=input_size,
                        out_features=output_size,
                        rram_interface=rram_interface,
                        **layer_kwargs
                    ),
                    nn.ReLU()
                )
            except ImportError:
                self.layer = None
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input array
            
        Returns:
            Output array
        """
        if self.layer_type == 'spiking':
            # Convert continuous inputs to spikes if needed
            if inputs.max() > 1.0:
                # Normalize to [0, 1] if values are too high
                inputs = np.clip(inputs, 0, 1)
            return self.layer.forward(inputs)
        elif self.layer_type == 'reservoir':
            return self.layer.forward(inputs)
        elif self.layer is not None:
            try:
                import torch
                with torch.no_grad():
                    inputs_tensor = torch.from_numpy(inputs).float()
                    if len(inputs_tensor.shape) == 1:
                        inputs_tensor = inputs_tensor.unsqueeze(0)  # Add batch dimension if needed
                    output_tensor = self.layer(inputs_tensor)
                    return output_tensor.squeeze(0).numpy()  # Remove batch dimension
            except ImportError:
                # Fallback if PyTorch is not available
                return inputs  # Identity in fallback
        else:
            return inputs  # Identity if no layer available
    
    def transform(self, inputs, **kwargs):
        """Transform method for pipeline integration."""
        return self.forward(inputs)


class RRAMPipeline:
    """
    Complete pipeline for RRAM-based computing with multiple components.
    """
    
    def __init__(self, 
                 components: Optional[List[RRAMPipelineComponent]] = None,
                 rram_interface: Optional[object] = None,
                 verbose: bool = False):
        """
        Initialize the RRAM pipeline.
        
        Args:
            components: List of pipeline components
            rram_interface: Optional RRAM interface
            verbose: Whether to print progress information
        """
        self.components = components or []
        self.rram_interface = rram_interface
        self.verbose = verbose
        self.history = []
    
    def add_component(self, component: RRAMPipelineComponent) -> 'RRAMPipeline':
        """
        Add a component to the pipeline.
        
        Args:
            component: RRAM pipeline component
            
        Returns:
            Self for method chaining
        """
        self.components.append(component)
        return self
    
    def fit(self, X, y=None, **kwargs):
        """
        Fit all components to data.
        
        Args:
            X: Input data
            y: Target data (optional)
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        current_data = X
        
        for i, component in enumerate(self.components):
            if self.verbose:
                print(f"Fitting component {i+1}/{len(self.components)}: {component.name}")
            
            start_time = time.time()
            
            if hasattr(component, 'fit'):
                component.fit(current_data, y, **kwargs)
            
            if hasattr(component, 'transform'):
                current_data = component.transform(current_data, **kwargs)
            
            end_time = time.time()
            
            self.history.append({
                'component': component.name,
                'fit_time': end_time - start_time,
                'step': i+1
            })
        
        return self
    
    def transform(self, X, **kwargs):
        """
        Transform data through all pipeline components.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Returns:
            Transformed data
        """
        current_data = X
        
        for i, component in enumerate(self.components):
            if self.verbose:
                print(f"Transforming with component {i+1}/{len(self.components)}: {component.name}")
            
            start_time = time.time()
            
            current_data = component.transform(current_data, **kwargs)
            
            end_time = time.time()
            
            self.history.append({
                'component': component.name,
                'transform_time': end_time - start_time,
                'step': i+1
            })
        
        return current_data
    
    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit and transform in one step.
        
        Args:
            X: Input data
            y: Target data (optional)
            **kwargs: Additional arguments
            
        Returns:
            Transformed data
        """
        return self.fit(X, y, **kwargs).transform(X, **kwargs)
    
    def predict(self, X, **kwargs):
        """
        Alias for transform method.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        return self.transform(X, **kwargs)
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray,
                metric: str = 'mse') -> Dict[str, float]:
        """
        Evaluate the pipeline performance.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            metric: Evaluation metric ('mse', 'mae', 'accuracy')
            
        Returns:
            Dictionary of evaluation results
        """
        predictions = self.predict(X_test)
        
        if metric == 'mse':
            error = np.mean((predictions - y_test) ** 2)
        elif metric == 'mae':
            error = np.mean(np.abs(predictions - y_test))
        elif metric == 'accuracy':
            # For classification or binary output
            accuracy = np.mean(np.isclose(predictions, y_test, rtol=0.1))
            return {'accuracy': float(accuracy)}
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return {metric: float(error)}


class RRAMDataPipelineIntegration:
    """
    Integration class for connecting RRAM systems to standard data science workflows.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 pipeline_configs: Optional[Dict[str, Any]] = None):
        """
        Initialize the data pipeline integration.
        
        Args:
            rram_interface: Optional RRAM interface
            pipeline_configs: Configuration for different pipeline types
        """
        self.rram_interface = rram_interface
        self.pipeline_configs = pipeline_configs or {}
        self.pipelines = {}
        self.data_loaders = {}
    
    def create_linear_system_pipeline(self, 
                                    name: str = 'default',
                                    fault_tolerance: bool = True,
                                    **config) -> RRAMPipeline:
        """
        Create a pipeline for solving linear systems with RRAM.
        
        Args:
            name: Name for the pipeline
            fault_tolerance: Whether to include fault tolerance
            **config: Additional configuration
            
        Returns:
            RRAMPipeline configured for linear systems
        """
        pipeline = RRAMPipeline(rram_interface=self.rram_interface, verbose=True)
        
        # Add transformers
        transformer = RRAMTransformer(
            rram_interface=self.rram_interface,
            normalize=True,
            scale_to_range=(1e-6, 1e-3)  # Typical RRAM conductance range
        )
        pipeline.add_component(transformer)
        
        # Add solver
        solver = RRAMLinearSystemSolver(
            solver_type=config.get('solver_type', 'hp_inv'),
            rram_interface=self.rram_interface,
            fault_tolerance=fault_tolerance,
            **config.get('solver_kwargs', {})
        )
        pipeline.add_component(solver)
        
        self.pipelines[name] = pipeline
        return pipeline
    
    def create_neural_pipeline(self,
                             name: str = 'neural',
                             layer_sizes: List[int] = None,
                             **config) -> RRAMPipeline:
        """
        Create a pipeline for neural network operations with RRAM.
        
        Args:
            name: Name for the pipeline
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            **config: Additional configuration
            
        Returns:
            RRAMPipeline configured for neural networks
        """
        if layer_sizes is None:
            layer_sizes = [10, 20, 15, 1]  # Default architecture
        
        pipeline = RRAMPipeline(rram_interface=self.rram_interface, verbose=True)
        
        # Add input transformer
        input_transformer = RRAMTransformer(
            rram_interface=self.rram_interface,
            normalize=True
        )
        pipeline.add_component(input_transformer)
        
        # Add neural layers
        layer_type = config.get('layer_type', 'spiking')
        for i in range(len(layer_sizes) - 1):
            layer = RRAMNeuralLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i+1],
                layer_type=layer_type,
                rram_interface=self.rram_interface
            )
            pipeline.add_component(layer)
        
        self.pipelines[name] = pipeline
        return pipeline
    
    def integrate_with_sklearn(self, pipeline_name: str = 'default'):
        """
        Create a wrapper to integrate with scikit-learn.
        
        Args:
            pipeline_name: Name of the pipeline to wrap
            
        Returns:
            Sklearn-compatible estimator
        """
        try:
            from sklearn.base import BaseEstimator, RegressorMixin
            from sklearn.utils.validation import check_X_y, check_array
            from sklearn.utils.multiclass import unique_labels
            
            class RRAMSklearnEstimator(BaseEstimator, RegressorMixin):
                def __init__(self, rram_integration, pipeline_name):
                    self.rram_integration = rram_integration
                    self.pipeline_name = pipeline_name
                    self.pipeline = rram_integration.pipelines[pipeline_name]
                
                def fit(self, X, y):
                    X, y = check_X_y(X, y)
                    self.pipeline.fit(X, y)
                    return self
                
                def predict(self, X):
                    X = check_array(X)
                    return self.pipeline.predict(X)
            
            return RRAMSklearnEstimator(self, pipeline_name)
        
        except ImportError:
            warnings.warn("Scikit-learn not available, skipping sklearn integration")
            return None
    
    def integrate_with_pandas(self):
        """
        Add RRAM methods to pandas DataFrame.
        """
        def rram_solve_system(df, b_col=None):
            """
            Solve linear system where DataFrame rows are matrix rows.
            """
            matrix = df.values
            
            if b_col is not None:
                b = df[b_col].values
                df = df.drop(columns=[b_col])
            else:
                b = np.ones(len(df))
            
            # Use RRAM to solve
            solver = RRAMLinearSystemSolver(rram_interface=self.rram_interface)
            solution, info = solver.solve_system(matrix, b)
            
            return solution
        
        # Add method to DataFrame class
        pd.DataFrame.rram_solve_system = rram_solve_system
    
    def benchmark_pipeline(self, 
                          pipeline_name: str,
                          test_data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                          num_iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark a pipeline's performance.
        
        Args:
            pipeline_name: Name of pipeline to benchmark
            test_data: Test data (input) or (input, output) tuple
            num_iterations: Number of iterations for averaging
            
        Returns:
            Benchmark results
        """
        pipeline = self.pipelines[pipeline_name]
        
        if isinstance(test_data, tuple) and len(test_data) == 2:
            X, y = test_data
        else:
            X = test_data
            y = None
        
        total_time = 0
        results = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            
            if y is not None:
                pred = pipeline.fit_transform(X, y)
            else:
                pred = pipeline.transform(X)
            
            end_time = time.time()
            total_time += (end_time - start_time)
            results.append(pred)
        
        avg_time = total_time / num_iterations
        
        benchmark_results = {
            'average_time_per_iteration': avg_time,
            'num_iterations': num_iterations,
            'total_time': total_time,
            'pipeline_name': pipeline_name,
            'timestamp': time.time()
        }
        
        # Add RRAM-specific metrics if interface is available
        if self.rram_interface is not None:
            # Get RRAM-specific metrics from the interface
            try:
                benchmark_results['rram_metrics'] = {
                    'hardware_connected': getattr(self.rram_interface, 'connected', False),
                    'last_operation_time': getattr(self.rram_interface, 'last_operation_time', 0)
                }
            except (AttributeError, RuntimeError, IOError) as e:
                warnings.warn(f"Failed to collect RRAM metrics from interface: {e}")
                pass
        
        return benchmark_results


def create_standard_rram_pipeline(task_type: str = 'linear_system',
                                 rram_interface: Optional[object] = None) -> RRAMPipeline:
    """
    Create a standard RRAM pipeline for common tasks.
    
    Args:
        task_type: Type of task ('linear_system', 'neural_network', 'classification')
        rram_interface: Optional RRAM interface
        
    Returns:
        Configured RRAM pipeline
    """
    if task_type == 'linear_system':
        # Create pipeline for solving linear systems
        pipeline = RRAMPipeline(rram_interface=rram_interface)
        
        # Add transformer
        transformer = RRAMTransformer(
            rram_interface=rram_interface,
            normalize=True,
            scale_to_range=(1e-6, 1e-3)
        )
        pipeline.add_component(transformer)
        
        # Add fault-tolerant solver
        solver = RRAMLinearSystemSolver(
            solver_type='hp_inv',
            rram_interface=rram_interface,
            fault_tolerance=True
        )
        pipeline.add_component(solver)
        
    elif task_type == 'neural_network':
        # Create pipeline for neural network operations
        pipeline = RRAMPipeline(rram_interface=rram_interface)
        
        # Add input transformer
        input_transformer = RRAMTransformer(
            rram_interface=rram_interface,
            normalize=True
        )
        pipeline.add_component(input_transformer)
        
        # Add spiking neural layer
        neural_layer = RRAMNeuralLayer(
            input_size=10,
            output_size=1,
            layer_type='spiking',
            rram_interface=rram_interface
        )
        pipeline.add_component(neural_layer)
        
    else:  # Default to linear system
        pipeline = create_standard_rram_pipeline('linear_system', rram_interface)
    
    return pipeline


def demonstrate_data_pipeline_integration():
    """
    Demonstrate the data pipeline integration capabilities.
    """
    print("Demonstrating RRAM Data Pipeline Integration...")
    
    # Create synthetic data
    n = 8
    G = np.random.rand(n, n) * 1e-4
    G = G + 0.5 * np.eye(n)  # Diagonally dominant
    b = np.random.rand(n)
    
    print(f"Created system: G (shape {G.shape}), b (shape {b.shape})")
    
    # Create a data pipeline
    pipeline_integration = RRAMDataPipelineIntegration()
    
    # Create a linear system pipeline
    linear_pipeline = pipeline_integration.create_linear_system_pipeline(
        name='linear_solver',
        fault_tolerance=True
    )
    
    print("Created linear system pipeline with fault tolerance")
    
    # Fit and transform data through the pipeline
    result = linear_pipeline.fit_transform((G, b))
    
    if isinstance(result, tuple):
        solution, info = result
        print(f"Pipeline solution successful")
        print(f"Solution residual: {np.linalg.norm(G @ solution - b):.2e}")
        print(f"Solution info: {info}")
    else:
        print(f"Pipeline result shape: {result.shape}")
    
    # Create a neural network pipeline
    neural_pipeline = pipeline_integration.create_neural_pipeline(
        name='neural',
        layer_sizes=[n, 15, 10, 1]
    )
    
    print("Created neural network pipeline")
    
    # Transform input through neural pipeline
    neural_input = np.random.rand(10)  # Input for neural network
    neural_output = neural_pipeline.transform(neural_input)
    print(f"Neural pipeline input shape: {neural_input.shape}")
    print(f"Neural pipeline output shape: {neural_output.shape}")
    
    # Benchmark the pipeline
    benchmark_results = pipeline_integration.benchmark_pipeline(
        'linear_solver',
        (G, b),
        num_iterations=5
    )
    
    print(f"Benchmark results: {benchmark_results}")
    
    print("Data pipeline integration demonstration completed!")


if __name__ == "__main__":
    demonstrate_data_pipeline_integration()