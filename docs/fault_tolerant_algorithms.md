# Fault-Tolerant Algorithms for RRAM Systems

## Overview

The `fault_tolerant_algorithms` module provides algorithms that can work around RRAM defects, degradation, and other reliability issues. This module implements various fault-tolerant techniques to ensure reliable computation even in the presence of hardware anomalies.

## Components

### FaultType Enum
Defines the types of faults that can occur in RRAM systems:
- `STUCK_AT_FAULT`: Conductance permanently stuck at high or low value
- `DRIFT`: Gradual change in conductance over time  
- `VARIABILITY`: Device-to-device variations
- `TDDB`: Time Dependent Dielectric Breakdown
- `THERMAL`: Temperature-related variations
- `AGING`: Long-term degradation

### FaultToleranceStrategy Enum
Defines strategies for handling faults:
- `REDUNDANCY`: Use redundant elements to replace faulty ones
- `RETRY`: Attempt operation multiple times with perturbations
- `ADAPTIVE`: Adjust parameters based on fault characteristics
- `MARGIN`: Use safety margins to account for uncertainty
- `ERROR_CORRECTION`: Apply error correction codes

### FaultDetector Class
Detects various types of faults in RRAM systems:
- Detects stuck-at faults based on conductance thresholds
- Identifies high variability in conductance values
- Monitors for drift by comparing measurements over time
- Analyzes matrix health for different fault types

### FaultTolerantHPINV Class
Fault-tolerant implementation of the HP-INV algorithm:
- Handles multiple fault types automatically
- Implements several recovery strategies
- Provides solution validation and confidence metrics
- Tracks recovery attempts and execution time

### AdaptiveFaultTolerance Class
Learns from system behavior to improve fault tolerance:
- Selects optimal strategies based on current conditions
- Updates strategy effectiveness based on outcomes
- Maintains performance history for learning

### FaultTolerantNeuralNetwork Class
Neural network implementation with built-in fault tolerance:
- Handles stuck weights and connections
- Applies redundancy to critical components
- Monitors for degradation over time

## Usage Examples

### Basic Fault-Tolerant Solving
```python
from fault_tolerant_algorithms import FaultTolerantHPINV, FaultToleranceStrategy

# Create fault-tolerant solver with redundancy strategy
ft_solver = FaultTolerantHPINV(
    fault_tolerance_strategy=FaultToleranceStrategy.REDUNDANCY,
    max_retries=3
)

# Solve linear system Gx = b with fault tolerance
result = ft_solver.solve(G, b, max_iter=20, tol=1e-6)

print(f"Solution successful: {result.success}")
print(f"Recovery attempts: {result.recovery_attempts}")
print(f"Residual norm: {result.error_metrics['residual_norm']:.2e}")
```

### Adaptive Fault Tolerance
```python
from fault_tolerant_algorithms import AdaptiveFaultTolerance

# Create adaptive system
adaptive_system = AdaptiveFaultTolerance()

# Solve adaptively based on system conditions
result = adaptive_system.solve_adaptively(G, b, max_iter=20, tol=1e-6)

print(f"Strategy used: {result.recovery_strategy.value}")
print(f"Adaptive solution successful: {result.success}")
```

## Key Features

1. **Comprehensive Fault Detection**: Identifies multiple types of RRAM faults
2. **Multiple Recovery Strategies**: Offers different approaches for handling faults
3. **Adaptive Learning**: Improves strategy selection over time
4. **Performance Monitoring**: Tracks success rates and efficiency
5. **Neural Network Integration**: Extends fault tolerance to neural computations
6. **Validation Metrics**: Provides confidence measures for solutions

## Benefits

- **Increased Reliability**: Maintains operation despite hardware defects
- **Adaptive Behavior**: Optimizes approach based on current conditions
- **Scalable Design**: Works for both small and large matrix operations
- **Performance Preservation**: Minimizes impact on computational efficiency
- **Integration Ready**: Compatible with existing RRAM infrastructure

This module significantly enhances the robustness of RRAM-based computing systems by implementing sophisticated fault tolerance mechanisms that can handle the inherent challenges of analog resistive memory devices.