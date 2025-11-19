# ML Framework Integration for RRAM Systems

## Overview

The `ml_framework_integration` module provides integration with popular ML frameworks (PyTorch and TensorFlow) for using RRAM simulations in neural network training and inference. This module enables neuromorphic computing applications using the RRAM simulator.

## Components

### RRAMLinearLayer Abstract Class
Abstract base class for RRAM-based linear layers:
- Defines interface for RRAM-aware neural operations
- Handles hardware simulation effects
- Manages weight quantization and non-idealities

### TorchRRAMLinear Class
RRAM-based linear layer for PyTorch:
- Implements RRAM simulation with PyTorch integration
- Applies conductance quantization and variability
- Supports hardware-in-the-loop simulation
- Handles stuck-at-fault effects

### TorchRRAMOptimizer Class
Optimization that accounts for RRAM non-idealities:
- Specialized optimizer for RRAM layers
- Adjusts learning rates for RRAM parameters
- Manages hardware simulation during training

### TensorFlowRRAMLayer Class
RRAM-based layer for TensorFlow/Keras:
- Implements RRAM effects in TensorFlow
- Handles quantization and variability
- Supports Keras integration

### RRAMModelWrapper Class
Tool to convert traditional models to RRAM-compatible:
- Converts standard models to RRAM equivalents
- Preserves model functionality
- Adds RRAM simulation effects

## Usage Examples

### Creating RRAM Neural Networks
```python
from ml_framework_integration import create_rram_neural_network

# Create a neural network with RRAM layers
network = create_rram_neural_network(
    input_size=784,           # MNIST input size
    hidden_sizes=[256, 128],  # Hidden layer sizes
    output_size=10,           # 10 classes for MNIST
    rram_interface=rram_hw,   # Optional hardware interface
    quantization_bits=4       # Weight quantization bits
)

print(network)
```

### Using RRAM Layers in PyTorch
```python
import torch
import torch.nn as nn
from ml_framework_integration import TorchRRAMLinear

# Create a network using RRAM layers
class RRAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TorchRRAMLinear(784, 256, quantization_bits=4)
        self.relu = nn.ReLU()
        self.layer2 = TorchRRAMLinear(256, 10, quantization_bits=4)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = RRAMNet()
print(model)
```

### Training with RRAM Simulation
```python
from ml_framework_integration import TorchRRAMOptimizer, train_with_rram_simulation

# Create optimizer that accounts for RRAM non-idealities
optimizer = TorchRRAMOptimizer(
    model,
    base_optimizer_class=torch.optim.Adam,
    base_lr=1e-3
)

# Train with RRAM simulation effects
train_with_rram_simulation(
    model=model,
    train_loader=train_loader,
    epochs=10,
    learning_rate=1e-3,
    framework='torch'
)
```

### Model Conversion
```python
from ml_framework_integration import RRAMModelWrapper

# Wrap an existing model to make it RRAM-compatible
wrapper = RRAMModelWrapper(original_model, framework='torch')
rram_model = wrapper.convert_to_rram_model(
    quantization_bits=6,
    rram_interface=rram_hw
)

# Benchmark the converted model
results = benchmark_rram_model(
    model=rram_model,
    test_loader=test_loader,
    framework='torch'
)
print(f"Accuracy: {results['accuracy']:.2f}%")
```

## Key Features

1. **Framework Integration**: PyTorch and TensorFlow compatibility
2. **RRAM Simulation**: Realistic simulation of RRAM non-idealities
3. **Quantization**: Automatic weight quantization to RRAM precision
4. **Hardware Effects**: Conductance variability, stuck faults, etc.
5. **Optimizer Support**: Specialized optimizers for RRAM networks
6. **Conversion Tools**: Convert existing models to RRAM equivalents
7. **Benchmarking**: Performance evaluation tools
8. **Hardware Interface**: Support for real hardware integration

## RRAM Effects Simulation

### Quantization
- Automatic weight quantization to specified bit precision
- Simulates digital-to-analog conversion limitations

### Variability
- Device-to-device conductance variations
- Process variation effects

### Stuck-at-Faults
- Permanent conductance failures
- Simulation of defective devices

### Line Resistance
- Interconnect resistance effects
- IR drop simulation

### Thermal Effects
- Temperature-dependent conductance
- Noise level adjustment

## PyTorch Integration

### RRAM Linear Layer
- Drop-in replacement for nn.Linear
- Maintains PyTorch autograd compatibility
- Supports backpropagation through RRAM simulation

### Optimizer
- Adjusts learning rates for RRAM parameters
- Accounts for quantization effects
- Maintains training stability

## TensorFlow Integration

### RRAM Layer
- Keras-compatible RRAM layer
- Handles TensorFlow gradients
- Integrates with Keras model API

## Benefits

- **Realistic Simulation**: Accurate modeling of RRAM effects
- **Framework Compatibility**: Works with standard ML frameworks
- **Easy Conversion**: Convert existing models with minimal changes
- **Performance Testing**: Evaluate RRAM feasibility for tasks
- **Hardware Ready**: Designed for physical RRAM integration
- **Backpropagation**: Maintains gradient flow for training
- **Quantization Awareness**: Proper handling of digital/analog interface
- **Scalability**: Supports large neural networks

## Use Cases

### Research Applications
- Evaluate RRAM for specific ML tasks
- Test algorithm compatibility
- Assess quantization impact

### Development
- Prototype RRAM neural accelerators
- Develop RRAM-aware algorithms
- Optimize networks for RRAM deployment

### Production
- Deploy models to RRAM hardware
- Create hybrid systems
- Optimize for edge devices

This ML framework integration enables researchers and developers to explore the potential of RRAM-based neural computing while maintaining compatibility with standard development practices.