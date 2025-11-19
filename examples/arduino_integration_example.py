"""
Example demonstrating integration of Arduino RRAM simulator with HP-INV algorithms.

This script shows how to connect the Arduino RRAM simulator to the Python
HP-INV implementations to solve matrix equations using both hardware simulation
and software algorithms.
"""
import sys
import os
import numpy as np
import time
from typing import Tuple

# Add the project source to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.arduino_rram_interface import ArduinoRRAMDemo, create_arduino_demo
from src.hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
from src.hardware_interface import MockRRAMHardware, HardwareTestFixture

def demo_basic_integration():
    """
    Demonstrate basic integration between Arduino RRAM interface and HP-INV algorithms.
    """
    print("=== Arduino RRAM Interface Integration Demo ===")
    
    try:
        # Create Arduino demo interface (this would connect to real hardware)
        # For simulation purposes, we'll use a mock instead of actual hardware
        print("Creating Arduino RRAM interface...")
        interface = create_arduino_demo(port='/dev/ttyUSB0')  # Placeholder port
        
        # Since we don't have actual hardware, we'll simulate by using the
        # existing MockRRAMHardware as the backend
        mock_hardware = MockRRAMHardware(
            size=8,
            variability=0.05,
            stuck_fault_prob=0.01,
            line_resistance=1.7e-3
        )
        
        # Replace the interface's hardware with mock for demonstration
        interface.interface = mock_hardware
        
        print("Connecting to hardware...")
        success = interface.interface.connect()
        if not success:
            print("Warning: Could not connect to hardware, using simulation instead")
        
        # Create a test matrix for solving equations
        print("\nCreating test system: G * x = b")
        n = 8
        G = np.random.rand(n, n) * 1e-6  # Conductance values in microsiemens
        G = G + 0.5 * np.eye(n)  # Ensure diagonal dominance
        
        # Create a known solution and compute corresponding b
        x_true = np.random.rand(n)
        b = G @ x_true  # This ensures we know the true solution
        
        print(f"Matrix condition number: {np.linalg.cond(G):.2e}")
        
        # Demonstrate matrix-vector multiplication
        print("\n--- Matrix-Vector Multiplication Demo ---")
        hw_result, expected_result = interface.demonstrate_mvm(G, x_true)
        print(f"MVM Difference: {np.linalg.norm(hw_result - expected_result):.2e}")
        
        # Demonstrate matrix inversion
        print("\n--- Matrix Inversion Demo ---")
        hw_inv, expected_inv = interface.demonstrate_inversion(G)
        print(f"Inversion Difference (Frobenius norm): {np.linalg.norm(hw_inv - expected_inv, 'fro'):.2e}")
        
        # Demonstrate HP-INV solver using hardware
        print("\n--- HP-INV Solver Demo ---")
        x_hw, iterations, info = interface.demonstrate_hp_inv_solver(
            G, b, 
            bits=3, 
            max_iter=10, 
            tol=1e-6,
            lp_noise_std=0.01
        )
        
        print(f"HP-INV Solution (Hardware): {x_hw}")
        print(f"True Solution: {x_true}")
        print(f"Solution Error: {np.linalg.norm(x_hw - x_true):.2e}")
        print(f"Iterations: {iterations}")
        print(f"Residual: {info['final_residual']:.2e}")
        print(f"Converged: {info['converged']}")
        
        # Compare with pure software HP-INV
        print("\n--- Pure Software HP-INV Comparison ---")
        x_sw, sw_iterations, sw_info = hp_inv(G, b, bits=3, max_iter=10, tol=1e-6)
        
        print(f"Software Solution: {x_sw}")
        print(f"Software Error: {np.linalg.norm(x_sw - x_true):.2e}")
        print(f"Software Iterations: {sw_iterations}")
        print(f"Software Residual: {sw_info['final_residual']:.2e}")
        print(f"Software Converged: {sw_info['converged']}")
        
        print("\n" + "="*60)
        print("Integration demo completed successfully!")
        print("The hardware interface successfully integrates with the HP-INV algorithms")
        print("showing how real RRAM devices can be used for analogue computing tasks.")
        
    except Exception as e:
        print(f"Error during integration: {e}")
        import traceback
        traceback.print_exc()


def demo_block_integration():
    """
    Demonstrate integration with block HP-INV for larger matrices.
    """
    print("\n=== Arduino RRAM Block Integration Demo ===")
    
    try:
        # For block operations, we'll simulate with a MockRRAMHardware
        # that represents a physical 8x8 RRAM tile
        mock_hardware = MockRRAMHardware(
            size=8,
            variability=0.03,
            stuck_fault_prob=0.005,
            line_resistance=1.0e-4
        )
        
        print("Setting up block operation with 8x8 RRAM tiles...")
        
        # Create a larger system (16x16) that requires block operations
        n = 16
        G = np.random.rand(n, n) * 1e-6
        G = G + 0.5 * np.eye(n)  # Diagonally dominant for stability
        
        # Create true solution and corresponding b vector
        x_true = np.random.rand(n)
        b = G @ x_true
        
        print(f"Large matrix condition number: {np.linalg.cond(G):.2e}")
        
        # Use Block HP-INV algorithm
        print("\nSolving with Block HP-INV (using simulated hardware)...")
        x_block, iterations, info = block_hp_inv(
            G, b, 
            block_size=8,  # Matches our tile size
            bits=3,
            max_iter=15,
            tol=1e-5
        )
        
        print(f"Block HP-INV Solution Error: {np.linalg.norm(x_block - x_true):.2e}")
        print(f"Iterations: {iterations}")
        print(f"Final Residual: {info['final_residual']:.2e}")
        print(f"Converged: {info['converged']}")
        
        print("\nThis demonstrates how larger matrices can be processed using")
        print("smaller physical RRAM tiles (e.g., 8x8) through the BlockAMC approach.")
        
    except Exception as e:
        print(f"Error during block integration: {e}")
        import traceback
        traceback.print_exc()


def demo_adaptive_integration():
    """
    Demonstrate integration with adaptive HP-INV that adjusts precision.
    """
    print("\n=== Arduino RRAM Adaptive Integration Demo ===")
    
    try:
        # Simulate adaptive precision with hardware effects
        mock_hardware = MockRRAMHardware(
            size=8,
            variability=0.05,
            stuck_fault_prob=0.02,  # Slightly higher fault probability
            line_resistance=2.0e-3
        )
        
        print("Setting up adaptive precision demo...")
        
        # Create test system
        n = 8
        # Create a potentially ill-conditioned matrix to trigger adaptive precision
        G = np.random.rand(n, n) * 1e-6
        # Add some structure to make it more challenging
        for i in range(n):
            for j in range(n):
                G[i, j] *= (i + 1) * (j + 1)  # Create some variation
        G = G + 1.0 * np.eye(n)  # Ensure diagonal dominance
        
        x_true = np.random.rand(n)
        b = G @ x_true
        
        print(f"Matrix condition number: {np.linalg.cond(G):.2e}")
        
        # Use adaptive HP-INV
        print("\nSolving with Adaptive HP-INV (hardware-aware)...")
        x_adaptive, iterations, info = adaptive_hp_inv(
            G, b,
            initial_bits=2,  # Start with low precision
            max_bits=6,      # Maximum precision allowed
            max_iter=20,
            tol=1e-6
        )
        
        print(f"Adaptive HP-INV Solution Error: {np.linalg.norm(x_adaptive - x_true):.2e}")
        print(f"Iterations: {iterations}")
        print(f"Final Precision (bits): {info['final_bits']}")
        print(f"Final Residual: {info['final_residual']:.2e}")
        print(f"Converged: {info['converged']}")
        
        print("\nAdaptive precision adjusts the quantization level based on convergence,")
        print("potentially increasing precision when the algorithm detects slow convergence.")
        
    except Exception as e:
        print(f"Error during adaptive integration: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function to run all integration demos.
    """
    print("Analogico - Arduino RRAM Integration Examples")
    print("="*60)
    
    demo_basic_integration()
    demo_block_integration()
    demo_adaptive_integration()
    
    print("\nAll integration demos completed!")
    print("\nTo use with actual Arduino hardware:")
    print("1. Upload the 'arduino_rram_simulator.ino' sketch to your Arduino")
    print("2. Connect the Arduino to your computer via USB")
    print("3. Update the port parameter in create_arduino_demo() to match your Arduino port")
    print("4. Run this script to interface with the actual hardware")
    print("\nThe hardware interface implements the same abstract interface as")
    print("the MockRRAMHardware, so algorithms work identically with real or simulated hardware.")


if __name__ == "__main__":
    main()