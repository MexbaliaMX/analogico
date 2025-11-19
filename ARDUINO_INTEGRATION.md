# Arduino Integration Guide for Analogico

This guide explains how to use Arduino hardware as an RRAM simulator for the Analogico project, demonstrating the HP-INV algorithm for solving matrix equations using resistive RAM crossbars.

## Overview

The Analogico project simulates Resistive RAM (RRAM) crossbar operations using Python, but can also interface with physical hardware. This implementation provides an interface to connect an Arduino-based RRAM simulator to the Python HP-INV implementations.

## Hardware Setup

### Components Required:
- Arduino board (Uno, Nano, Mega, etc.)
- Electronic components to simulate RRAM behavior (resistors, potentiometers, or specialized RRAM emulator circuitry)
- USB cable for Arduino connection

### Arduino Sketch Installation

1. Install the Arduino IDE from https://www.arduino.cc/en/software
2. Install required libraries:
   - ArduinoJson library (for JSON serialization/deserialization)
   
3. Upload the `arduino_rram_simulator.ino` sketch to your Arduino:
   - Open the Arduino IDE
   - Create a new sketch
   - Copy and paste the contents of `arduino_rram_simulator.ino` file
   - Select your Arduino board and COM port
   - Upload the sketch

## Software Installation

Install the required Python dependencies:

```bash
pip install pyserial
```

The Arduino interface is part of the Analogico project and uses the same dependencies as the main project.

## Usage

### Basic Interface Example

```python
from src.arduino_rram_interface import ArduinoRRAMDemo, create_arduino_demo

# Create the Arduino interface (adjust port as needed)
demo = create_arduino_demo(port='/dev/ttyUSB0')  # Linux
# demo = create_arduino_demo(port='COM3')        # Windows

# Connect to the Arduino
demo.interface.connect()

# Create a test matrix
import numpy as np
test_matrix = np.random.rand(8, 8) * 1e-6  # Conductance values
test_vector = np.random.rand(8)

# Perform matrix-vector multiplication on the Arduino
hw_result, expected_result = demo.demonstrate_mvm(test_matrix, test_vector)

# Perform matrix inversion on the Arduino
hw_inv, expected_inv = demo.demonstrate_inversion(test_matrix)

# Use the HP-INV solver with Arduino assistance
b = test_matrix @ np.ones(8)  # Create known solution
x_hw, iterations, info = demo.demonstrate_hp_inv_solver(
    test_matrix, b, 
    bits=3, 
    max_iter=10, 
    tol=1e-6
)

# Disconnect from Arduino
demo.interface.disconnect()
```

### Integration with HP-INV Algorithms

The Arduino interface can be used with the main HP-INV algorithms:

```python
from src.hp_inv import hp_inv
from src.arduino_rram_interface import ArduinoRRAMDemo
from src.hardware_interface import MockRRAMHardware

# Use with mock hardware for testing without actual Arduino
mock_hardware = MockRRAMHardware(
    size=8,
    variability=0.05,
    stuck_fault_prob=0.01,
    line_resistance=1.7e-3
)

# Create demo interface using mock hardware as an example
demo = ArduinoRRAMDemo(mock_hardware)

# Connect the hardware
mock_hardware.connect()

# Create test system
G = np.random.rand(8, 8) * 0.1 + np.eye(8)  # Well-conditioned
b = np.random.rand(8)

# Run the hardware-assisted solver
x_hw, iterations, info = demo.demonstrate_hp_inv_solver(
    G, b,
    bits=3,
    max_iter=15,
    tol=1e-6
)

print(f"Solution found in {iterations} iterations")
print(f"Final residual: {info['final_residual']:.2e}")
```

## Protocol

The communication between Python and Arduino uses a JSON-based serial protocol:

- Commands are sent as JSON strings followed by a newline
- Responses are received as JSON strings with a newline
- All operations are blocking (synchronous)

Example command: `{"cmd": "MVM", "vector": [1.0, 2.0, 3.0, 4.0]}`
Example response: `{"result": [0.98, 2.01, 3.02, 4.01]}`

## Available Commands

- `INIT`: Initialize the RRAM simulator
- `CONFIG`: Configure device parameters (variability, fault probability, line resistance)
- `WRITE_MATRIX`: Program a conductance matrix to the RRAM crossbar
- `READ_MATRIX`: Read the current conductance matrix from the RRAM crossbar
- `MVM`: Perform matrix-vector multiplication
- `INVERT`: Perform matrix inversion

## Running Examples

To run the Arduino integration example:

```bash
python examples/arduino_integration_example.py
```

## Testing

Run the integration tests to verify functionality:

```bash
python tests/test_arduino_integration.py
```

## Troubleshooting

1. **Serial port not found**: Ensure the Arduino is connected and the correct port is specified. Use `ls /dev/tty*` (Linux/Mac) or check Device Manager (Windows) to identify the port.

2. **Upload errors**: Make sure the ArduinoJson library is installed in the Arduino IDE.

3. **Communication errors**: Check baud rate settings match between Python and Arduino code.

## Limitations

- Performance will be slower than pure software simulation due to serial communication overhead
- Matrix sizes are limited by Arduino memory constraints
- The simulation in the Arduino is simplified compared to real RRAM behavior

## Real Hardware Connection

To connect to real RRAM hardware instead of a simulator:

1. Modify the Arduino sketch to interface with actual RRAM components
2. Update the simulation parameters to match your specific RRAM device characteristics
3. Ensure proper voltage and current levels for your RRAM array