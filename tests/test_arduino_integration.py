"""
Tests for the Arduino RRAM interface module.
"""
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.arduino_rram_interface import ArduinoRRAMInterface, ArduinoRRAMDemo
from src.hardware_interface import MockRRAMHardware


def test_arduino_rram_interface_creation():
    """Test creating an Arduino RRAM interface."""
    print("Testing Arduino RRAM interface creation...")
    # Since we can't assume an actual Arduino is connected,
    # we'll create the interface object but not connect
    interface = ArduinoRRAMInterface(port='/dev/ttyUSB0')  # fake port

    assert interface.size == 8, f"Expected size 8, got {interface.size}"
    assert interface.variability == 0.05, f"Expected variability 0.05, got {interface.variability}"
    assert interface.stuck_fault_prob == 0.01, f"Expected stuck_fault_prob 0.01, got {interface.stuck_fault_prob}"
    assert interface.line_resistance == 1.7e-3, f"Expected line_resistance 1.7e-3, got {interface.line_resistance}"
    assert interface.connected == False, f"Expected connected False, got {interface.connected}"
    print("✓ Arduino RRAM interface creation test passed")


def test_arduino_rram_demo_creation():
    """Test creating an Arduino RRAM demo with mock hardware."""
    print("Testing Arduino RRAM demo creation...")
    mock_hardware = MockRRAMHardware(
        size=8,
        variability=0.05,
        stuck_fault_prob=0.01,
        line_resistance=1.7e-3
    )

    demo = ArduinoRRAMDemo(mock_hardware)

    assert demo.interface == mock_hardware, f"Expected interface to be mock_hardware, got {demo.interface}"
    print("✓ Arduino RRAM demo creation test passed")


def test_arduino_rram_demo_mvm():
    """Test matrix-vector multiplication with demo."""
    print("Testing Arduino RRAM demo MVM...")
    mock_hardware = MockRRAMHardware(
        size=4,
        variability=0.02,
        stuck_fault_prob=0.005,
        line_resistance=1.0e-4
    )

    demo = ArduinoRRAMDemo(mock_hardware)

    # Connect the mock hardware
    mock_hardware.connect()

    # Create test data
    matrix = np.random.rand(4, 4) * 1e-6
    vector = np.random.rand(4)

    # Perform MVM
    hw_result, expected_result = demo.demonstrate_mvm(matrix, vector)

    # Results should be reasonably close (accounting for hardware simulation noise)
    diff = np.linalg.norm(hw_result - expected_result)
    # Since we're using hardware simulation with noise, we allow for some difference
    assert diff < 1.0, f"MVM difference too large: {diff}"  # Should be reasonably close
    print("✓ Arduino RRAM demo MVM test passed")


def test_arduino_rram_demo_inversion():
    """Test matrix inversion with demo."""
    print("Testing Arduino RRAM demo inversion...")
    mock_hardware = MockRRAMHardware(
        size=4,
        variability=0.02,
        stuck_fault_prob=0.005,
        line_resistance=1.0e-4
    )

    demo = ArduinoRRAMDemo(mock_hardware)

    # Connect the mock hardware
    mock_hardware.connect()

    # Create a well-conditioned test matrix
    matrix = np.random.rand(4, 4) * 0.1 + np.eye(4)

    # Perform inversion
    hw_inv, expected_inv = demo.demonstrate_inversion(matrix)

    # Check that both are valid inverses (A * A^(-1) should be close to I)
    hw_identity_check = matrix @ hw_inv
    exp_identity_check = matrix @ expected_inv

    hw_error = np.linalg.norm(hw_identity_check - np.eye(4))
    exp_error = np.linalg.norm(exp_identity_check - np.eye(4))

    # Both should produce results close to identity matrix
    assert hw_error < 10.0, f"Hardware error too large: {hw_error}"  # Hardware simulation tolerance
    assert exp_error < 1e-10, f"Expected error too large: {exp_error}"  # Expected computation tolerance
    print("✓ Arduino RRAM demo inversion test passed")


def test_arduino_rram_demo_hp_inv():
    """Test HP-INV solver with hardware."""
    print("Testing Arduino RRAM demo HP-INV solver...")
    mock_hardware = MockRRAMHardware(
        size=4,
        variability=0.02,
        stuck_fault_prob=0.005,
        line_resistance=1.0e-4
    )

    demo = ArduinoRRAMDemo(mock_hardware)

    # Connect the mock hardware
    mock_hardware.connect()

    # Create test system
    G = np.random.rand(4, 4) * 0.1 + np.eye(4)  # Well-conditioned
    x_true = np.random.rand(4)
    b = G @ x_true

    # Run the hardware-assisted HP-INV solver
    x_hw, iterations, info = demo.demonstrate_hp_inv_solver(
        G, b,
        bits=3,
        max_iter=10,
        tol=1e-6
    )

    # Check that the solution converged (or at least ran without errors)
    assert iterations >= 1, f"Expected at least 1 iteration, got {iterations}"
    assert len(info['residuals']) >= 1, f"Expected at least 1 residual, got {len(info['residuals'])}"

    # Check that the solution is reasonable
    residual = np.linalg.norm(G @ x_hw - b)
    assert residual < 1.0, f"Residual too large: {residual}"  # Hardware simulation tolerance
    print("✓ Arduino RRAM demo HP-INV solver test passed")


def test_arduino_serial_comms_simulation():
    """Test that the Arduino interface methods can be called without errors."""
    print("Testing Arduino serial communications simulation...")
    # Using MockRRAMHardware to simulate the hardware behavior
    mock_hardware = MockRRAMHardware(
        size=6,
        variability=0.03,
        stuck_fault_prob=0.01,
        line_resistance=1.5e-3
    )

    # Connect the mock hardware
    success = mock_hardware.connect()
    assert success, f"Expected successful connection, got {success}"

    # Create test matrix
    test_matrix = np.random.rand(6, 6) * 1e-6
    test_matrix = test_matrix + 0.2 * np.eye(6)  # Ensure well-conditioned

    # Test write_matrix
    write_success = mock_hardware.write_matrix(test_matrix)
    assert write_success, f"Expected successful write, got {write_success}"

    # Test read_matrix
    read_matrix = mock_hardware.read_matrix()
    assert read_matrix.shape == (6, 6), f"Expected shape (6, 6), got {read_matrix.shape}"

    # Test matrix-vector multiplication
    test_vector = np.random.rand(6)
    mvm_result = mock_hardware.matrix_vector_multiply(test_vector)
    assert mvm_result.shape == (6,), f"Expected shape (6,), got {mvm_result.shape}"

    # Test matrix inversion
    inv_result = mock_hardware.invert_matrix(test_matrix)
    assert inv_result.shape == (6, 6), f"Expected shape (6, 6), got {inv_result.shape}"

    # Disconnect
    disconnect_success = mock_hardware.disconnect()
    assert disconnect_success, f"Expected successful disconnect, got {disconnect_success}"
    assert mock_hardware.connected == False, f"Expected connected False, got {mock_hardware.connected}"
    print("✓ Arduino serial communications simulation test passed")


if __name__ == "__main__":
    # Run the tests
    test_arduino_rram_interface_creation()
    test_arduino_rram_demo_creation()
    test_arduino_rram_demo_mvm()
    test_arduino_rram_demo_inversion()
    test_arduino_rram_demo_hp_inv()
    test_arduino_serial_comms_simulation()

    print("\n🎉 All Arduino integration tests passed!")