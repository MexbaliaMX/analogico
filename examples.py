"""
Usage Examples for Analogico Project

This file contains comprehensive usage examples for the Analogico project APIs.
"""
import numpy as np
from src.hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv, blockamc_inversion
from src.rram_model import create_rram_matrix, mvm
from src.redundancy import apply_redundancy
from src.stress_test import run_stress_test
from src.hardware_interface import MockRRAMHardware, HardwareTestFixture


def example_1_basic_hp_inv():
    """
    Example 1: Basic HP-INV usage
    """
    print("=== Example 1: Basic HP-INV ===")
    
    # Create a test system
    G = np.random.rand(4, 4) + 0.5 * np.eye(4)  # Well-conditioned matrix
    b = np.random.rand(4)
    
    # Solve using HP-INV
    x, iters, info = hp_inv(G, b, bits=4, max_iter=20, tol=1e-8)
    
    print(f"Solution found in {iters} iterations")
    print(f"Final residual: {info['final_residual']:.2e}")
    print(f"Converged: {info['converged']}")
    print()


def example_2_rram_model():
    """
    Example 2: Using RRAM model with realistic properties
    """
    print("=== Example 2: RRAM Model ===")
    
    # Create an RRAM matrix with realistic parameters
    G = create_rram_matrix(
        n=6,
        variability=0.04,        # 4% variability
        stuck_fault_prob=0.02,   # 2% stuck faults
        line_resistance=2.0e-3   # 2 mΩ line resistance
    )
    
    print(f"Matrix shape: {G.shape}")
    print(f"Number of stuck faults: {len(G[G == 0])}")
    print(f"Conductance range: {G.min():.2e} - {G.max():.2e} S")
    
    # Perform matrix-vector multiplication
    x = np.ones(6)
    y = mvm(G, x)
    print(f"MVM result: {y}")
    print()


def example_3_with_redundancy():
    """
    Example 3: Using redundancy to repair faults
    """
    print("=== Example 3: Redundancy ===")
    
    # Create matrix with stuck faults
    G_with_faults = create_rram_matrix(6, stuck_fault_prob=0.05)
    print(f"Before repair: {len(G_with_faults[G_with_faults == 0])} stuck faults")
    
    # Apply redundancy
    G_repaired = apply_redundancy(G_with_faults)
    print(f"After repair: {len(G_repaired[G_repaired == 0])} stuck faults remaining")
    
    # Ensure the matrix is well-conditioned and add diagonal dominance
    G_repaired = G_repaired + 0.3 * np.eye(6)
    
    # Check that the repair worked
    b = np.random.rand(6)
    x, iters, info = hp_inv(G_repaired, b, max_iter=15, tol=1e-6)
    print(f"HP-INV worked: {iters} iterations, converged: {info['converged']}")
    print()


def example_4_block_hp_inv():
    """
    Example 4: Using Block HP-INV for large matrices
    """
    print("=== Example 4: Block HP-INV ===")
    
    # Create a larger RRAM matrix
    G_large = create_rram_matrix(16, variability=0.03, stuck_fault_prob=0.01)
    G_large = G_large + 0.3 * np.eye(16)  # Ensure well-conditioned
    b_large = np.random.rand(16)
    
    # Solve using Block HP-INV
    x_block, iters_block, info_block = block_hp_inv(G_large, b_large, block_size=8)
    print(f"Block HP-INV completed in {iters_block} iterations")
    print(f"Final residual: {info_block['final_residual']:.2e}")
    print(f"Converged: {info_block['converged']}")
    print()


def example_5_adaptive_hp_inv():
    """
    Example 5: Using Adaptive HP-INV
    """
    print("=== Example 5: Adaptive HP-INV ===")
    
    G = np.random.rand(6, 6) + 0.5 * np.eye(6)  # Smaller matrix for stability
    b = np.random.rand(6)
    
    x_adaptive, iters, info = adaptive_hp_inv(G, b, initial_bits=2, max_bits=4, max_iter=20)
    print(f"Adaptive HP-INV used {info['final_bits']} bits and {iters} iterations")
    print(f"Final residual: {info['final_residual']:.2e}")
    print(f"Converged: {info['converged']}")
    print()


def example_6_blockamc_inversion():
    """
    Example 6: BlockAMC Matrix Inversion
    """
    print("=== Example 6: BlockAMC Inversion ===")
    
    # Use a smaller, well-conditioned matrix for better results
    G = np.random.rand(8, 8) * 0.1 + np.eye(8)  # Conditioned matrix
    G_inv = blockamc_inversion(G, block_size=4)
    
    # Verify the inverse
    identity_result = G @ G_inv
    error = np.linalg.norm(identity_result - np.eye(8))
    print(f"Inversion error: {error:.2e}")
    print()


def example_7_stress_test():
    """
    Example 7: Running Stress Tests
    """
    print("=== Example 7: Stress Testing ===")
    
    # Run a small stress test
    iters, errors = run_stress_test(
        n=6,
        num_samples=30,  # Small number for quick test
        variability=0.03,
        stuck_prob=0.02,
        bits=4
    )
    
    print(f"Ran {len(iters)} successful tests out of 30")
    print(f"Average iterations: {np.mean(iters):.2f}")
    print(f"Average error: {np.mean(errors):.2e}")
    print(f"Max error: {np.max(errors):.2e}")
    print()


def example_8_hardware_interface():
    """
    Example 8: Hardware Interface Simulation
    """
    print("=== Example 8: Hardware Interface ===")
    
    # Create mock hardware
    hardware = MockRRAMHardware(size=4, variability=0.02, stuck_fault_prob=0.01)
    
    # Connect and program a matrix
    hardware.connect()
    test_matrix = np.random.rand(4, 4) * 1e-6
    test_matrix = test_matrix + 0.5 * np.eye(4)
    
    success = hardware.write_matrix(test_matrix)
    if success:
        print("Matrix successfully programmed to hardware")
        
        # Perform MVM
        vector = np.ones(4)
        result = hardware.matrix_vector_multiply(vector)
        print(f"MVM result: {result}")
        
        # Perform inversion
        inv_result = hardware.invert_matrix(test_matrix)
        print(f"Inversion completed")
    
    hardware.disconnect()
    print()


def example_9_complete_workflow():
    """
    Example 9: Complete Workflow Example
    """
    print("=== Example 9: Complete Workflow ===")
    
    # Step 1: Create an RRAM matrix with realistic properties
    print("Creating RRAM matrix with realistic properties...")
    G = create_rram_matrix(
        n=8,
        variability=0.03,      # 3% conductance variability
        stuck_fault_prob=0.01, # 1% stuck-at faults
        line_resistance=1.5e-3 # 1.5 mΩ line resistance
    )
    
    # Step 2: Apply redundancy to mitigate stuck faults
    print("Applying redundancy to repair faults...")
    G_repaired = apply_redundancy(G)
    # Ensure the matrix is well-conditioned for inversion
    G_repaired = G_repaired + 0.3 * np.eye(8)
    
    # Step 3: Generate right-hand side vector
    b = np.random.randn(8)
    
    # Step 4: Solve using HP-INV
    print("Solving linear system using HP-INV...")
    x_solution, iterations, info = hp_inv(
        G_repaired,
        b,
        bits=4,              # Use 4-bit precision
        lp_noise_std=0.01,   # 1% noise in low-precision inversion
        max_iter=15,         # Allow up to 15 iterations
        tol=1e-6,            # Convergence tolerance
        relaxation_factor=1.0 # Relaxation factor for convergence
    )
    
    # Step 5: Validate the solution
    residual = np.linalg.norm(G_repaired @ x_solution - b)
    print(f"Solution found in {iterations} iterations")
    print(f"Final residual: {residual:.2e}")
    print(f"Converged: {info['converged']}")
    print(f"Condition number: {info['condition_number_estimate']:.2e}")
    print()


def example_10_comparing_methods():
    """
    Example 10: Comparing Different HP-INV Methods
    """
    print("=== Example 10: Comparing HP-INV Methods ===")

    # Create test system
    G = create_rram_matrix(8, variability=0.02, stuck_fault_prob=0.005)
    G = G + 0.4 * np.eye(8)  # Ensure well-conditioned
    G = apply_redundancy(G)
    b = np.random.randn(8)

    # Solve with different approaches
    methods = [
        ("Standard HP-INV", lambda: hp_inv(G, b, max_iter=20)),
        ("Block HP-INV", lambda: block_hp_inv(G, b, block_size=4, max_iter=20)),
        ("Adaptive HP-INV", lambda: adaptive_hp_inv(G, b, initial_bits=2, max_iter=20))
    ]

    for name, method in methods:
        x, iters, info = method()
        residual = np.linalg.norm(G @ x - b)
        print(f"{name}: {iters} iterations, residual {residual:.2e}")

    print()


def example_11_arduino_integration():
    """
    Example 11: Arduino RRAM Integration
    Demonstrates integration with Arduino-based RRAM simulator
    """
    print("=== Example 11: Arduino RRAM Integration ===")

    try:
        from src.arduino_rram_interface import ArduinoRRAMDemo, create_arduino_demo
        from src.hardware_interface import MockRRAMHardware

        # Since we don't have actual hardware in this example,
        # we'll simulate using MockRRAMHardware
        mock_hardware = MockRRAMHardware(
            size=8,
            variability=0.05,
            stuck_fault_prob=0.01,
            line_resistance=1.7e-3
        )

        # Create demo interface using mock hardware
        demo = ArduinoRRAMDemo(mock_hardware)

        print("Arduino RRAM interface created with mock hardware backend")

        # Create test matrices
        test_matrix = np.random.rand(8, 8) * 1e-6  # Conductance values
        test_matrix = test_matrix + 0.3 * np.eye(8)  # Ensure well-conditioned
        test_vector = np.random.rand(8)

        # Demonstrate matrix-vector multiplication
        hw_result, expected_result = demo.demonstrate_mvm(test_matrix, test_vector)
        print(f"MVM difference: {np.linalg.norm(hw_result - expected_result):.2e}")

        # Demonstrate matrix inversion
        hw_inv, expected_inv = demo.demonstrate_inversion(test_matrix)
        print(f"Inversion difference: {np.linalg.norm(hw_inv - expected_inv, 'fro'):.2e}")

        # Demonstrate HP-INV solver integration
        b = test_matrix @ np.ones(8)  # Create known solution
        x_hw, iterations, info = demo.demonstrate_hp_inv_solver(
            test_matrix, b,
            bits=3,
            max_iter=10,
            tol=1e-6
        )
        print(f"HP-INV converged in {iterations} iterations with residual {info['final_residual']:.2e}")

        print("Arduino integration example completed successfully!")

    except ImportError as e:
        print(f"Could not run Arduino integration example: {e}")
        print("Make sure you have all required dependencies installed.")
    print()


def main():
    """
    Run all examples
    """
    example_1_basic_hp_inv()
    example_2_rram_model()
    example_3_with_redundancy()
    example_4_block_hp_inv()
    example_5_adaptive_hp_inv()
    example_6_blockamc_inversion()
    example_7_stress_test()
    example_8_hardware_interface()
    example_9_complete_workflow()
    example_10_comparing_methods()
    example_11_arduino_integration()

    print("All examples completed!")

if __name__ == "__main__":
    main()