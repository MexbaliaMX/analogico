# Neuromorphic Computing Modules for RRAM Systems

## Overview

The `neuromorphic_modules` module implements neuromorphic computing approaches that take advantage of RRAM's analog computing capabilities, including spiking neural networks, online learning, and other brain-inspired algorithms. This module leverages the physics of RRAM devices to implement biologically-inspired computing paradigms.

## Components

### SpikingNeuron Class
Spiking neuron model simulating RRAM behavior:
- Leaky integrate-and-fire dynamics
- Refractory period modeling
- RRAM-inspired resistance switching
- Temporal dynamics simulation

### RRAMSpikingLayer Class
Layer of spiking neurons using RRAM for synaptic weights:
- Synaptic weight implementation with RRAM
- Spike-Timing-Dependent Plasticity (STDP)
- Spike history tracking
- Quantized weight updates

### RRAMReservoirComputer Class
Reservoir computer using RRAM for the reservoir layer:
- Complex nonlinear transformations
- Echo state property maintenance
- Quantized reservoir weights
- Online learning capabilities

### RRAMHebbianLearner Class
Hebbian learning using RRAM's analog properties:
- "Hebbian learning: neurons that fire together, wire together"
- RRAM-based correlation storage
- Online weight updates
- Biological learning rule implementation

### RRAMNeuromorphicNetwork Class
Complete neuromorphic network with RRAM components:
- Multiple layer types integration
- Flexible network architectures
- Hardware simulation capabilities
- Performance optimization

### RRAMSynapticPlasticity Class
Synaptic plasticity mechanisms using RRAM:
- Spike-Timing-Dependent Plasticity (STDP)
- Homeostatic regulation
- Long-term potentiation/depression
- Synaptic strength updates

## Usage Examples

### Creating Spiking Networks
```python
from neuromorphic_modules import RRAMSpikingLayer, SpikingNeuron

# Create a spiking layer
spiking_layer = RRAMSpikingLayer(
    n_inputs=100,
    n_neurons=50,
    rram_interface=rram_hw,
    quantization_bits=4
)

# Simulate spiking activity
input_spikes = (np.random.random(100) > 0.8).astype(float)
output_spikes = spiking_layer.forward(input_spikes)

print(f"Input spikes: {np.sum(input_spikes)}")
print(f"Output spikes: {np.sum(output_spikes)}")

# Update weights using STDP
spiking_layer.update_weights_stdp(learning_rate=0.01)
```

### Reservoir Computing
```python
from neuromorphic_modules import RRAMReservoirComputer

# Create reservoir computer
reservoir = RRAMReservoirComputer(
    input_size=20,
    reservoir_size=200,
    output_size=5,
    spectral_radius=0.9,
    connectivity=0.1,
    rram_interface=rram_hw
)

# Train the reservoir
inputs_seq = [np.random.randn(20) for _ in range(50)]
targets_seq = [np.random.randn(5) for _ in range(50)]
reservoir.train_output([inputs_seq], [targets_seq])

# Use the reservoir
input_vector = np.random.randn(20)
output = reservoir.forward(input_vector)
print(f"Reservoir output: {output}")
```

### Creating Neuromorphic Networks
```python
from neuromorphic_modules import RRAMNeuromorphicNetwork

# Define network configuration
layers_config = [
    {
        'type': 'spiking',
        'n_inputs': 784,      # 28x28 image
        'n_neurons': 256,
        'quantization_bits': 4
    },
    {
        'type': 'reservoir',
        'input_size': 256,
        'reservoir_size': 500,
        'output_size': 10,    # 10 classes
        'quantization_bits': 4
    }
]

# Create the network
network = RRAMNeuromorphicNetwork(layers_config, rram_interface=rram_hw)

# Process input
input_data = np.random.randn(784)
output = network.forward(input_data)
print(f"Network output shape: {output.shape}")
```

### Hebbian Learning
```python
from neuromorphic_modules import RRAMHebbianLearner

# Create Hebbian learner
hebbian = RRAMHebbianLearner(
    num_synapses=100,
    rram_interface=rram_hw,
    learning_rate=0.01
)

# Update weights based on neural activity
pre_activities = np.random.random(100)
post_activities = np.random.random(100)
hebbian.update_weights_hebbian(pre_activities, post_activities)

# Compute correlation
pattern1 = np.random.random(100)
pattern2 = np.random.random(100)
correlation = hebbian.correlate(pattern1, pattern2)
print(f"Pattern correlation: {correlation:.4f}")
```

### Autoencoder Network
```python
from neuromorphic_modules import create_neuromorphic_autoencoder

# Create neuromorphic autoencoder
autoencoder = create_neuromorphic_autoencoder(
    input_size=784,
    hidden_size=128,
    rram_interface=rram_hw
)

# Use for encoding/decoding
input_data = np.random.randn(784)
encoded = autoencoder.forward(input_data)
print(f"Autoencoder output shape: {encoded.shape}")
```

## Key Features

1. **Spiking Neural Networks**: Event-driven computation with temporal dynamics
2. **Reservoir Computing**: Complex nonlinear transformations with fixed reservoir
3. **Hebbian Learning**: Biologically-inspired correlation-based learning
4. **Synaptic Plasticity**: STDP and homeostatic regulation mechanisms
5. **RRAM Integration**: Direct mapping to RRAM physical properties
6. **Quantization Aware**: Optimized for RRAM precision constraints
7. **Online Learning**: Continuous adaptation capability
8. **Hardware Ready**: Designed for RRAM hardware implementation

## Neuromorphic Principles

### Spiking Neural Networks
- Information encoded in spike timing
- Event-driven, power-efficient computation
- Temporal processing capabilities
- Analog computation in synapses

### Reservoir Computing
- Fixed random reservoir for nonlinear mapping
- Only output layer is trained
- Echo state property for stability
- Rich representation of temporal patterns

### Hebbian Learning
- Synaptic strength based on neural correlation
- "What fires together, wires together"
- Unsupervised learning
- Local learning rules

### Synaptic Plasticity
- Long-term potentiation/depression
- Homeostatic regulation
- Biological learning mechanisms
- Continuous adaptation

## Benefits

- **Energy Efficiency**: Event-driven spiking computation
- **Temporal Processing**: Natural handling of temporal patterns
- **Biological Plausibility**: Neuroscience-inspired algorithms
- **RRAM Optimized**: Direct mapping to physical properties
- **Online Learning**: Continuous adaptation to new data
- **Scalability**: Efficient implementation for large networks
- **Robustness**: Inherent fault tolerance
- **Real-time Processing**: Natural parallel processing

## Algorithms Implemented

### Spike-Timing-Dependent Plasticity (STDP)
- Adjusts synaptic weights based on spike timing
- Biological learning mechanism
- Unsupervised correlation learning

### Reservoir Computing
- Fixed random recurrent network
- Linear readout layer training
- Rich computational capacity

### Homeostatic Regulation
- Maintains stable neural activity
- Prevents runaway excitation
- Adaptive threshold adjustment

## Applications

### Pattern Recognition
- Temporal pattern recognition
- Sequence learning
- Real-time classification

### Signal Processing
- Time-series prediction
- Filtering and transformation
- Adaptive processing

### Learning Systems
- Unsupervised learning
- Online adaptation
- Association and memory

This neuromorphic computing module provides a comprehensive toolkit for implementing brain-inspired algorithms using RRAM technology, enabling efficient and powerful neural computation.