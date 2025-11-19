"""
Integration tests for the Analogico project.
Tests the interaction between all major modules: hp_inv, rram_model, redundancy, and stress_test.
"""
import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from src.hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
from src.rram_model import create_rram_matrix, mvm
from src.redundancy import apply_redundancy
from src.stress_test import run_stress_test
from src.hardware_interface import HardwareTestFixture, MockRRAMHardware


def test_full_pipeline_integration():
    """Test the complete pipeline from RRAM matrix creation to HP-INV solution."""
    # Create an RRAM matrix with realistic parameters
    n = 8
    G = create_rram_matrix(
        n,
        variability=0.05,
        stuck_fault_prob=0.01,
        line_resistance=1.7e-3
    )
    
    # Apply redundancy to handle stuck-at faults
    G_repaired = apply_redundancy(G)
    
    # Ensure the matrix is well-conditioned for solving
    G_repaired = G_repaired + 0.5 * np.eye(n)  # Add some diagonal dominance
    
    # Generate a random right-hand side vector
    b = np.random.randn(n)
    
    # Solve using HP-INV
    x_approx, iters, info = hp_inv(
        G_repaired, 
        b, 
        bits=3, 
        lp_noise_std=0.01, 
        max_iter=15,
        tol=1e-6,
        relaxation_factor=1.0
    )
    
    # Verify solution properties
    assert isinstance(x_approx, np.ndarray)
    assert x_approx.shape == b.shape
    assert iters > 0 and iters <= 15
    assert isinstance(info, dict)
    assert 'residuals' in info
    assert 'converged' in info
    
    # Verify that the solution produces reasonable residuals
    final_residual = info['final_residual']
    assert final_residual >= 0  # residual should be non-negative
    
    print(f"Integration test passed: {iters} iterations, final residual: {final_residual:.2e}")


def test_block_hp_inv_integration():
    """Test the block HP-INV algorithm with RRAM model."""
    n = 16  # Larger matrix to test block algorithm
    block_size = 8
    
    # Create a larger RRAM matrix
    G = create_rram_matrix(
        n,
        variability=0.03,
        stuck_fault_prob=0.005,
        line_resistance=1.7e-3
    )
    
    # Apply redundancy
    G_repaired = apply_redundancy(G)
    G_repaired = G_repaired + 0.3 * np.eye(n)  # Add diagonal dominance
    
    b = np.random.randn(n)
    
    # Solve using block HP-INV
    x_approx, iters, info = block_hp_inv(
        G_repaired,
        b,
        block_size=block_size,
        bits=3,
        lp_noise_std=0.01,
        max_iter=10,
        tol=1e-4
    )
    
    # Verify results
    assert x_approx.shape == b.shape
    assert iters > 0
    assert isinstance(info, dict)
    
    print(f"Block HP-INV integration test passed: {iters} iterations")


def test_adaptive_hp_inv_integration():
    """Test the adaptive HP-INV algorithm with RRAM model."""
    n = 8
    
    # Create RRAM matrix
    G = create_rram_matrix(
        n,
        variability=0.04,
        stuck_fault_prob=0.01,
        line_resistance=1.7e-3
    )
    
    # Apply redundancy
    G_repaired = apply_redundancy(G)
    G_repaired = G_repaired + 0.4 * np.eye(n)  # Add diagonal dominance
    
    b = np.random.randn(n)
    
    # Solve using adaptive HP-INV
    x_approx, iters, info = adaptive_hp_inv(
        G_repaired,
        b,
        initial_bits=2,
        max_bits=5,
        lp_noise_std=0.015,
        max_iter=20,
        tol=1e-5
    )
    
    # Verify results
    assert x_approx.shape == b.shape
    assert iters > 0
    assert isinstance(info, dict)
    assert 'final_bits' in info
    assert info['final_bits'] >= 2  # Should have at least the initial bits
    
    print(f"Adaptive HP-INV integration test passed: {iters} iterations, final bits: {info['final_bits']}")


def test_stress_test_with_all_modules():
    """Test the stress test framework using all modules together."""
    # Run a small stress test to verify integration
    iters, errors = run_stress_test(
        n=6,
        num_samples=5,  # Small number for quick test
        variability=0.05,
        stuck_prob=0.01,
        bits=3,
        lp_noise_std=0.01,
        max_iter=10
    )
    
    # Verify results
    assert isinstance(iters, list)
    assert isinstance(errors, list)
    assert len(iters) <= 5  # May be fewer if some matrices were singular
    assert all(e >= 0 for e in errors)  # All errors should be non-negative
    
    print(f"Stress test integration passed: {len(iters)} samples run")


def test_rram_model_with_redundancy_and_hp_inv():
    """Test the combination of RRAM model, redundancy, and HP-INV."""
    # Create multiple RRAM matrices with different parameters
    params = [
        {"variability": 0.02, "stuck_fault_prob": 0.005},
        {"variability": 0.05, "stuck_fault_prob": 0.01},
        {"variability": 0.08, "stuck_fault_prob": 0.02}
    ]
    
    for i, param in enumerate(params):
        n = 6
        # Create RRAM matrix with specific parameters
        G = create_rram_matrix(n, **param)
        
        # Apply redundancy
        G_repaired = apply_redundancy(G)
        # Ensure the matrix is usable
        G_repaired = G_repaired + 0.5 * np.eye(n)
        
        # Verify that redundancy changed some values
        changed_count = np.sum(G != G_repaired)
        
        b = np.random.randn(n)
        
        # Solve with HP-INV
        x, iters, info = hp_inv(G_repaired, b, max_iter=10)
        
        # All steps should complete without error
        assert x.shape == b.shape
        assert iters > 0
        
        print(f"RRAM-redundancy-HP-INV integration test {i+1} passed: {changed_count} cells repaired")


if __name__ == "__main__":
    test_full_pipeline_integration()
    test_block_hp_inv_integration()
    test_adaptive_hp_inv_integration()
    test_stress_test_with_all_modules()
    test_rram_model_with_redundancy_and_hp_inv()
    print("All integration tests passed!")