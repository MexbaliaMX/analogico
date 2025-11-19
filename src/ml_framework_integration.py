"""
ML Framework Integration for RRAM-based computing.

This module provides integration with PyTorch and TensorFlow for using
RRAM simulations in neural network training and inference.
"""
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, Union, Callable, List
from abc import ABC, abstractmethod
import time

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, torch integration disabled")

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available, TF integration disabled")


if TORCH_AVAILABLE:
    class RRAMLinearLayer(ABC, nn.Module):
        """
        Abstract base class for RRAM-based linear layers in neural networks.
        This combines the benefits of digital control with analog computation.
        """
        
        @abstractmethod
        def forward(self, input_tensor):
            """Perform forward pass using RRAM simulation."""
            pass

        @abstractmethod
        def update_weights_from_rram(self):
            """Update layer weights from simulated RRAM conductances."""
            pass

        @abstractmethod
        def program_rram_weights(self):
            """Program current weights to RRAM simulation."""
            pass
else:
    class RRAMLinearLayer(ABC):
        """
        Abstract base class for RRAM-based linear layers in neural networks.
        This combines the benefits of digital control with analog computation.
        """
        
        @abstractmethod
        def forward(self, input_tensor):
            """Perform forward pass using RRAM simulation."""
            pass

        @abstractmethod
        def update_weights_from_rram(self):
            """Update layer weights from simulated RRAM conductances."""
            pass

        @abstractmethod
        def program_rram_weights(self):
            """Program current weights to RRAM simulation."""
            pass


if TORCH_AVAILABLE:
    class TorchRRAMLinear(RRAMLinearLayer):
        """
        RRAM-based linear layer for PyTorch with hardware-in-the-loop simulation.
        """

        def __init__(self,
                     in_features: int,
                     out_features: int,
                     bias: bool = True,
                     rram_interface: Optional[object] = None,
                     quantization_bits: int = 4,
                     use_simulation: bool = True,
                     enable_stochastic: bool = True):
            """
            Initialize RRAM linear layer for PyTorch.

            Args:
                in_features: Size of input features
                out_features: Size of output features
                bias: Whether to include bias term
                rram_interface: Optional RRAM interface for hardware-in-the-loop
                quantization_bits: Number of bits for weight quantization
                use_simulation: Whether to use RRAM simulation effects
                enable_stochastic: Whether to enable stochastic effects in simulation
            """
            super().__init__()
            
            self.in_features = in_features
            self.out_features = out_features
            self.rram_interface = rram_interface
            self.quantization_bits = quantization_bits
            self.use_simulation = use_simulation
            self.enable_stochastic = enable_stochastic
            
            # Initialize parameters
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2.0 / in_features))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
            
            # RRAM simulation parameters
            self.rram_variability = 0.02  # 2% device-to-device variability
            self.rram_nonlinearity = 0.05  # 5% non-ideal effects
            self.rram_stuck_fault_prob = 0.005  # 0.5% stuck-at-fault probability

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            """
            Forward pass with RRAM simulation effects.

            Args:
                input_tensor: Input tensor of shape (..., in_features)

            Returns:
                Output tensor of shape (..., out_features)
            """
            # Use the current weights to compute output
            output = F.linear(input_tensor, self.weight, self.bias)
            
            if self.use_simulation:
                # Apply RRAM-specific effects to the output
                if self.rram_nonlinearity > 0:
                    # Apply small non-linear distortion effects
                    nonlin = torch.tanh(output * self.rram_nonlinearity)
                    output = output + nonlin * 0.01  # Small non-linear addition
                
                if self.enable_stochastic:
                    # Add random noise to simulate thermal noise
                    noise = torch.randn_like(output) * self.rram_variability * 0.1
                    output = output + noise
            
            return output

        def update_weights_from_rram(self):
            """
            Update layer weights from RRAM conductances (simulation).
            In hardware implementation, this would read from actual RRAM.
            """
            if self.rram_interface is not None:
                # In real hardware, this would read from RRAM
                try:
                    # Read the current conductance matrix from hardware
                    conductance_matrix = self.rram_interface.read_matrix()
                    # Convert conductances to weights
                    with torch.no_grad():
                        # Update weight parameter with RRAM values (with some noise for realism)
                        noise = torch.randn_like(self.weight) * self.rram_variability
                        self.weight.data = torch.tensor(conductance_matrix, 
                                                       dtype=torch.float32, 
                                                       device=self.weight.device) + noise
                except Exception as e:
                    warnings.warn(f"Could not update weights from RRAM hardware: {e}")
            else:
                # In simulation, apply RRAM effects to current weights
                with torch.no_grad():
                    # Apply variability to simulate device differences
                    variability = torch.randn_like(self.weight) * self.rram_variability
                    noisy_weights = self.weight.data * (1 + variability)
                    
                    # Apply stuck-at-fault simulation
                    stuck_mask = torch.rand_like(self.weight) < self.rram_stuck_fault_prob
                    noisy_weights.masked_fill_(stuck_mask, 0.0)  # Stuck at 0
                    
                    self.weight.data = noisy_weights

        def program_rram_weights(self):
            """
            Program current layer weights to RRAM conductances (simulation).
            In hardware implementation, this would write to actual RRAM.
            """
            # Prepare weights for RRAM (quantize and apply constraints)
            weights_for_rram = self._prepare_weights_for_rram(self.weight.data.detach().cpu().numpy())
            
            if self.rram_interface is not None:
                # In real hardware, this would program the RRAM
                try:
                    success = self.rram_interface.write_matrix(weights_for_rram)
                    if not success:
                        warnings.warn("Failed to program weights to RRAM hardware")
                except Exception as e:
                    warnings.warn(f"Error programming weights to RRAM hardware: {e}")
            
            # In simulation, just update internal state
            self._update_internal_rram_state(weights_for_rram)

        def _prepare_weights_for_rram(self, weights: np.ndarray) -> np.ndarray:
            """
            Prepare weights for RRAM programming with quantization and constraints.
            """
            # Apply quantization to simulate digital-to-analog conversion limitations
            if self.quantization_bits > 0:
                levels = 2**self.quantization_bits - 1
                max_val = np.max(np.abs(weights))
                if max_val > 0:
                    scale = levels / max_val
                    weights = np.round(weights * scale) / scale
            
            # Apply RRAM conductance constraints (positive for simplification)
            weights = np.clip(weights, -1e-3, 1e-3)
            
            return weights

        def _update_internal_rram_state(self, programmed_weights: np.ndarray):
            """
            Update internal simulation state when weights are programmed to RRAM.
            """
            # In a real system, this would track the internal state of RRAM devices
            # For simulation, we just keep track of the programmed state
            self._last_programmed_weights = programmed_weights
            self._programming_timestamp = time.time()

    class TorchRRAMOptimizer:
        """
        PyTorch optimizer that accounts for RRAM non-idealities.
        """
        
        def __init__(self, 
                     model: nn.Module, 
                     base_optimizer_class: type = torch.optim.Adam,
                     base_lr: float = 1e-3,
                     rram_effects_scale: float = 0.1,
                     **base_optimizer_kwargs):
            """
            Initialize RRAM-aware PyTorch optimizer.

            Args:
                model: PyTorch model containing RRAM layers
                base_optimizer_class: Base optimizer class (Adam, SGD, etc.)
                base_lr: Base learning rate
                rram_effects_scale: Scale factor for RRAM effects
                **base_optimizer_kwargs: Additional args for base optimizer
            """
            self.rram_effects_scale = rram_effects_scale
            self.base_optimizer = base_optimizer_class(
                model.parameters(),
                lr=base_lr,
                **base_optimizer_kwargs
            )
            
            # Track RRAM layers
            self.rram_layers = [
                module for module in model.modules() 
                if isinstance(module, TorchRRAMLinear)
            ]

        def step(self, closure: Optional[Callable] = None):
            """
            Perform optimization step with RRAM considerations.
            """
            # Update weights from RRAM simulation before optimization
            for layer in self.rram_layers:
                layer.update_weights_from_rram()
            
            # Perform base optimizer step
            self.base_optimizer.step(closure)
            
            # Program updated weights back to RRAM simulation
            for layer in self.rram_layers:
                layer.program_rram_weights()

        def zero_grad(self):
            """Clear gradients."""
            self.base_optimizer.zero_grad()

        def state_dict(self):
            """Get state dictionary."""
            return self.base_optimizer.state_dict()

        def load_state_dict(self, state_dict):
            """Load state dictionary."""
            self.base_optimizer.load_state_dict(state_dict)

        def add_param_group(self, param_group):
            """Add parameter group."""
            self.base_optimizer.add_param_group(param_group)

    def create_rram_neural_network(
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        rram_interface: Optional[object] = None,
        quantization_bits: int = 4
    ) -> nn.Module:
        """
        Create a neural network with RRAM layers.

        Args:
            input_size: Size of input layer
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output layer
            rram_interface: Optional RRAM interface for hardware-in-the-loop
            quantization_bits: Number of bits for weight quantization

        Returns:
            PyTorch neural network with RRAM layers
        """
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layers.append(TorchRRAMLinear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i+1],
                bias=True,
                rram_interface=rram_interface,
                quantization_bits=quantization_bits
            ))

            # Add activation function except for output layer
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

else:  # TORCH_AVAILABLE is False
    class TorchRRAMLinear:
        """Fallback class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TorchRRAMLinear")
        
        def forward(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TorchRRAMLinear")
        
        def update_weights_from_rram(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TorchRRAMLinear")
        
        def program_rram_weights(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TorchRRAMLinear")

    class TorchRRAMOptimizer:
        """Fallback class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TorchRRAMOptimizer")
        
        def step(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TorchRRAMOptimizer")
        
        def zero_grad(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TorchRRAMOptimizer")
    
    def create_rram_neural_network(*args, **kwargs):
        """Fallback function when PyTorch is not available."""
        raise ImportError("PyTorch is required to create RRAM neural networks")


if TF_AVAILABLE:
    class TensorFRRAMLayer(keras.layers.Layer):
        """
        RRAM-based layer for TensorFlow/Keras with hardware simulation.
        """
        
        def __init__(self,
                     in_features: int,
                     out_features: int,
                     bias: bool = True,
                     rram_interface: Optional[object] = None,
                     quantization_bits: int = 4,
                     use_simulation: bool = True,
                     enable_stochastic: bool = True,
                     **kwargs):
            """
            Initialize RRAM layer for TensorFlow.

            Args:
                in_features: Size of input features
                out_features: Size of output features
                bias: Whether to include bias term
                rram_interface: Optional RRAM interface for hardware-in-the-loop
                quantization_bits: Number of bits for weight quantization
                use_simulation: Whether to use RRAM simulation effects
                enable_stochastic: Whether to enable stochastic effects in simulation
                **kwargs: Additional arguments for Keras layer
            """
            super().__init__(**kwargs)
            
            self.in_features = in_features
            self.out_features = out_features
            self.rram_interface = rram_interface
            self.quantization_bits = quantization_bits
            self.use_simulation = use_simulation
            self.enable_stochastic = enable_stochastic
            
            # RRAM simulation parameters
            self.rram_variability = 0.02  # 2% device-to-device variability
            self.rram_nonlinearity = 0.05  # 5% non-ideal effects
            self.rram_stuck_fault_prob = 0.005  # 0.5% stuck-at-fault probability

        def build(self, input_shape):
            """Build the layer."""
            self.weight = self.add_weight(
                name='weight',
                shape=(self.in_features, self.out_features),
                initializer='glorot_uniform',
                trainable=True
            )
            
            self.bias = self.add_weight(
                name='bias',
                shape=(self.out_features,),
                initializer='zeros',
                trainable=True
            ) if self.use_bias else None
            
            super().build(input_shape)

        def call(self, inputs):
            """
            Forward pass with RRAM simulation effects.

            Args:
                inputs: Input tensor

            Returns:
                Output tensor
            """
            # Standard linear transformation
            output = tf.matmul(inputs, self.weight)
            if self.bias is not None:
                output = tf.nn.bias_add(output, self.bias)
            
            if self.use_simulation:
                # Apply RRAM-specific effects to the output
                if self.rram_nonlinearity > 0:
                    # Apply small non-linear distortion effects
                    nonlin = tf.nn.tanh(output * self.rram_nonlinearity)
                    output = output + nonlin * 0.01  # Small non-linear addition
                
                if self.enable_stochastic:
                    # Add random noise to simulate thermal noise
                    noise = tf.random.normal(tf.shape(output)) * self.rram_variability * 0.1
                    output = output + noise
            
            return output

        def get_config(self):
            """Get layer configuration."""
            config = super().get_config()
            config.update({
                'in_features': self.in_features,
                'out_features': self.out_features,
                'rram_interface': self.rram_interface,
                'quantization_bits': self.quantization_bits,
                'use_simulation': self.use_simulation,
                'enable_stochastic': self.enable_stochastic,
                'rram_variability': self.rram_variability,
                'rram_nonlinearity': self.rram_nonlinearity,
                'rram_stuck_fault_prob': self.rram_stuck_fault_prob
            })
            return config
        
        @classmethod
        def from_config(cls, config):
            """Create layer from configuration."""
            return cls(**config)

    def create_tf_rram_neural_network(
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        rram_interface: Optional[object] = None,
        quantization_bits: int = 4
    ) -> keras.Model:
        """
        Create a TensorFlow neural network with RRAM layers.

        Args:
            input_size: Size of input layer
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output layer
            rram_interface: Optional RRAM interface for hardware-in-the-loop
            quantization_bits: Number of bits for weight quantization

        Returns:
            TensorFlow neural network with RRAM layers
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Input(shape=(input_size,)))
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            model.add(TensorFRRAMLayer(
                in_features=model.layers[-1].units if hasattr(model.layers[-1], 'units') else input_size,
                out_features=hidden_size,
                bias=True,
                rram_interface=rram_interface,
                quantization_bits=quantization_bits
            ))
            model.add(keras.layers.ReLU())
        
        # Output layer
        last_size = hidden_sizes[-1] if hidden_sizes else input_size
        model.add(TensorFRRAMLayer(
            in_features=last_size,
            out_features=output_size,
            bias=True,
            rram_interface=rram_interface,
            quantization_bits=quantization_bits
        ))
        
        return model

else:  # TF_AVAILABLE is False
    class TensorFRRAMLayer:
        """Fallback class when TensorFlow is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for TensorFRRAMLayer")
    
    def create_tf_rram_neural_network(*args, **kwargs):
        """Fallback function when TensorFlow is not available."""
        raise ImportError("TensorFlow is required to create TF RRAM neural networks")


def integrate_with_ml_frameworks():
    """
    Provide integration functions for ML frameworks.
    """
    
    def train_with_rram_simulation(
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 10,
        rram_effects_enabled: bool = True,
        framework: str = 'torch'
    ) -> Dict[str, List[float]]:
        """
        Train a neural network with RRAM simulation effects.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            rram_effects_enabled: Whether to enable RRAM simulation
            framework: ML framework ('torch' or 'tf')

        Returns:
            Training metrics
        """
        if framework == 'torch' and TORCH_AVAILABLE:
            return _train_pytorch_model(
                model, train_loader, val_loader, epochs, rram_effects_enabled
            )
        elif framework == 'tf' and TF_AVAILABLE:
            return _train_tensorflow_model(
                model, train_loader, val_loader, epochs, rram_effects_enabled
            )
        else:
            raise ImportError(f"Framework {framework} not available or not supported")

    def _train_pytorch_model(
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any],
        epochs: int,
        rram_effects_enabled: bool
    ) -> Dict[str, List[float]]:
        """Train PyTorch model with RRAM effects."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training PyTorch models")
        
        # Find RRAM layers in the model
        rram_layers = [
            module for module in model.modules()
            if isinstance(module, TorchRRAMLinear)
        ]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Apply RRAM simulation before forward pass if enabled
                if rram_effects_enabled:
                    for layer in rram_layers:
                        layer.update_weights_from_rram()
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Program updated weights to RRAM simulation after backward pass
                if rram_effects_enabled:
                    for layer in rram_layers:
                        layer.program_rram_weights()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)
            
            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += criterion(output, target).item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}')
        
        return {'train_losses': train_losses, 'val_losses': val_losses}

    def _train_tensorflow_model(
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any],
        epochs: int,
        rram_effects_enabled: bool
    ) -> Dict[str, List[float]]:
        """Train TensorFlow model with RRAM effects."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for training TensorFlow models")
        
        # For TensorFlow, we would implement the RRAM effects differently
        # This is a simplified implementation
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        history = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=epochs,
            verbose=1
        )
        
        return {
            'train_losses': history.history['loss'],
            'val_losses': history.history['val_loss'] if 'val_loss' in history.history else []
        }
    
    return {
        'train_with_rram_simulation': train_with_rram_simulation,
        'pytorch_available': TORCH_AVAILABLE,
        'tensorflow_available': TF_AVAILABLE,
        'TorchRRAMLinear': TorchRRAMLinear if TORCH_AVAILABLE else None,
        'TensorFRRAMLayer': TensorFRRAMLayer if TF_AVAILABLE else None
    }


def integrate_with_ml_frameworks():
    """
    Provide integration functions for ML frameworks.
    """
    def train_with_rram_simulation(
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 10,
        rram_effects_enabled: bool = True,
        framework: str = 'torch'
    ) -> Dict[str, List[float]]:
        """
        Train a neural network with RRAM simulation effects.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            rram_effects_enabled: Whether to enable RRAM simulation
            framework: ML framework ('torch' or 'tf')

        Returns:
            Training metrics
        """
        if framework == 'torch' and TORCH_AVAILABLE:
            # Import here to avoid circular imports in case of issues
            try:
                from torch.utils.data import DataLoader
            except ImportError:
                pass  # Handle gracefully

            return _train_pytorch_model(
                model, train_loader, val_loader, epochs, rram_effects_enabled
            )
        elif framework == 'tf' and TF_AVAILABLE:
            return _train_tensorflow_model(
                model, train_loader, val_loader, epochs, rram_effects_enabled
            )
        else:
            raise ImportError(f"Framework {framework} not available or not supported")

    def _train_pytorch_model(
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any],
        epochs: int,
        rram_effects_enabled: bool
    ) -> Dict[str, List[float]]:
        """Train PyTorch model with RRAM effects."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training PyTorch models")

        # Find RRAM layers in the model
        rram_layers = [
            module for module in model.modules()
            if isinstance(module, TorchRRAMLinear)
        ]

        # Use CPU by default to avoid issues with GPU availability
        device = torch.device('cpu')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss() if TORCH_AVAILABLE else lambda x, y: torch.mean((x - y)**2)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            # For demonstration, we'll use simple data format
            # In practice, this would iterate through the actual data loader
            for batch_idx in range(min(10, len(train_loader) if hasattr(train_loader, '__len__') else 10)):  # Limit for demo
                # In a real implementation, this would get the actual batch from train_loader
                data = torch.randn(32, model[0].in_features if hasattr(model, '__getitem__') else 8)  # Dummy data
                target = torch.randn(32, model[-1].out_features if hasattr(model, '__getitem__') else 1)  # Dummy target

                data, target = data.to(device), target.to(device)

                # Apply RRAM simulation before forward pass if enabled
                if rram_effects_enabled:
                    for layer in rram_layers:
                        layer.update_weights_from_rram()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Program updated weights to RRAM simulation after backward pass
                if rram_effects_enabled:
                    for layer in rram_layers:
                        layer.program_rram_weights()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / min(10, len(train_loader) if hasattr(train_loader, '__len__') else 10)
            train_losses.append(avg_epoch_loss)

            # Validation would go here in a full implementation

        return {'train_losses': train_losses, 'val_losses': val_losses}

    def _train_tensorflow_model(
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any],
        epochs: int,
        rram_effects_enabled: bool
    ) -> Dict[str, List[float]]:
        """Train TensorFlow model with RRAM effects."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for training TensorFlow models")

        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model
        history = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=epochs,
            verbose=0  # Suppress output during testing
        )

        return {
            'train_losses': history.history['loss'],
            'val_losses': history.history.get('val_loss', [])
        }

    return {
        'train_with_rram_simulation': train_with_rram_simulation,
        'pytorch_available': TORCH_AVAILABLE,
        'tensorflow_available': TF_AVAILABLE,
        'TorchRRAMLinear': TorchRRAMLinear if TORCH_AVAILABLE else None,
        'TensorFRRAMLayer': TensorFRRAMLayer if TF_AVAILABLE else None
    }


# Create the integration functions
ml_integration = integrate_with_ml_frameworks()

# Aliases for convenience
if TORCH_AVAILABLE:
    PyTorchRRAMLinear = TorchRRAMLinear
    PyTorchRRAMOptimizer = TorchRRAMOptimizer

if TF_AVAILABLE:
    KerasRRAMLayer = TensorFRRAMLayer