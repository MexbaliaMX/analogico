# Data Pipeline Integration for RRAM Systems

## Overview

The `data_pipeline_integration` module provides tools for integrating RRAM computations into data science and machine learning pipelines. This module includes data loaders, transformers, pipeline components, and integration tools that make it easy to incorporate RRAM computing into standard workflows.

## Components

### RRAMDataLoader Class
Data loader specialized for RRAM operations:
- Loads matrices from various formats (NPY, NPZ, CSV, JSON)
- Creates synthetic datasets with controlled properties
- Handles linear system pairs (G, b) for solving
- Supports batch loading for large datasets

### RRAMTransformer Class
Transformer for preprocessing data for RRAM operations:
- Normalizes data to appropriate ranges
- Scales values to RRAM conductance ranges
- Handles the mapping between real values and conductance
- Maintains statistical information for inverse transforms

### RRAMPipelineComponent Base Class
Base class for pipeline components:
- Provides consistent interface across components
- Supports fit/transform pattern
- Enables method chaining
- Tracks component performance

### RRAMLinearSystemSolver Class
Pipeline component for solving linear systems with RRAM:
- Supports HP-INV, Block HP-INV, and Adaptive HP-INV
- Includes fault tolerance capabilities
- Handles (G, b) pairs for solving Gx=b
- Provides solution validation and metrics

### RRAMNeuralLayer Class
Pipeline component for RRAM-based neural networks:
- Supports spiking, reservoir, and linear layers
- Integrates with hardware interfaces
- Handles different layer architectures
- Manages neural activity patterns

### RRAMPipeline Class
Complete pipeline system for chaining operations:
- Manages multiple components in sequence
- Supports fit/transform/predict methods
- Tracks execution history and performance
- Provides evaluation capabilities

### RRAMDataPipelineIntegration Class
Main integration class that connects to standard workflows:
- Creates pre-configured pipelines for common tasks
- Integrates with pandas DataFrames
- Provides scikit-learn compatibility
- Includes benchmarking and evaluation tools

## Usage Examples

### Basic Data Loading and Transforming
```python
from data_pipeline_integration import RRAMDataLoader, RRAMTransformer

# Load matrix data from file
loader = RRAMDataLoader(batch_size=32)
for matrix, target in loader.load_matrix_data('data.npz'):
    print(f"Matrix shape: {matrix.shape}")

# Transform data for RRAM
transformer = RRAMTransformer(
    normalize=True, 
    scale_to_range=(1e-6, 1e-3)  # RRAM conductance range
)
transformed_data = transformer.fit_transform(raw_data)
```

### Creating and Using Pipelines
```python
from data_pipeline_integration import RRAMPipeline, RRAMLinearSystemSolver

# Create a simple pipeline
pipeline = RRAMPipeline()
pipeline.add_component(RRAMTransformer(normalize=True))
pipeline.add_component(RRAMLinearSystemSolver(fault_tolerance=True))

# Solve a linear system through the pipeline
solution, info = pipeline.fit_transform((G, b))
print(f"Solution residual: {info['error_metrics']['residual_norm']:.2e}")
```

### Integration with Pandas
```python
from data_pipeline_integration import RRAMDataPipelineIntegration
import pandas as pd

# Integrate with pandas
integration = RRAMDataPipelineIntegration()
integration.integrate_with_pandas()

# Now DataFrames have RRAM methods
df = pd.read_csv('matrix_data.csv')
solution = df.rram_solve_system(b_values)
```

### Creating Specialized Pipelines
```python
from data_pipeline_integration import RRAMDataPipelineIntegration

# Create a system for linear solving
integration = RRAMDataPipelineIntegration()
linear_pipeline = integration.create_linear_system_pipeline(
    name='linear_solver',
    fault_tolerance=True
)

# Create a system for neural networks
neural_pipeline = integration.create_neural_pipeline(
    name='neural',
    layer_sizes=[784, 256, 128, 10]
)
```

## Key Features

1. **Multiple Data Formats**: Supports NPY, NPZ, CSV, and JSON formats
2. **Pipeline Architecture**: Flexible component chaining
3. **Standard Integration**: Compatible with pandas and scikit-learn
4. **RRAM-Optimized**: Preprocessing specifically for analog computing
5. **Fault Tolerance**: Built-in reliability features
6. **Benchmarking**: Performance measurement tools
7. **Synthetic Data Generation**: Tools for creating test datasets
8. **Evaluation Metrics**: Comprehensive performance assessment

## Benefits

- **Workflow Integration**: Seamlessly integrates with existing data science workflows
- **RRAM Optimization**: Data preprocessing optimized for analog computing
- **Scalability**: Handles both small and large datasets
- **Reliability**: Includes fault tolerance and validation
- **Flexibility**: Configurable components for different use cases
- **Performance**: Efficient batch processing and caching
- **Standard Compatibility**: Works with scikit-learn, pandas, and other libraries
- **Extensible**: Easy to add new components and features

## Common Use Cases

### Scientific Computing
- Solving linear systems of equations
- Matrix operations for simulations
- Numerical analysis tasks

### Machine Learning
- Neural network inference with RRAM
- Feature transformation
- Model acceleration

### Data Processing
- Large-scale matrix computations
- Signal processing applications
- Optimization problems

This module bridges the gap between traditional digital computing workflows and RRAM-based analog computing, enabling researchers and developers to leverage the efficiency of RRAM systems while maintaining compatibility with standard data science tools.