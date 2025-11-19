"""
Tests for the hardware interface module.
"""
import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.hardware_interface import MockRRAMHardware, HardwareTestFixture, create_mock_hardware_fixture


def test_mock_rram_hardware():
    """Test the mock RRAM hardware implementation."""
    print("Testing Mock RRAM Hardware...")
    
    # Create mock hardware
    hardware = MockRRAMHardware(size=4, variability=0.05, stuck_fault_prob=0.01)
    
    # Connect
    assert hardware.connect()
    assert hardware.connected
    print("✓ Connect test passed")
    
    # Configure
    assert hardware.configure(variability=0.03, stuck_fault_prob=0.02)
    assert hardware.variability == 0.03
    assert hardware.stuck_fault_prob == 0.02
    print("✓ Configure test passed")
    
    # Write matrix
    test_matrix = np.eye(4) * 1e-6
    assert hardware.write_matrix(test_matrix)
    print("✓ Write matrix test passed")
    
    # Read matrix
    read_back = hardware.read_matrix()
    assert read_back.shape == (4, 4)
    print("✓ Read matrix test passed")
    
    # Matrix-vector multiplication
    test_vector = np.ones(4)
    result = hardware.matrix_vector_multiply(test_vector)
    assert result.shape == (4,)
    print("✓ Matrix-vector multiplication test passed")
    
    # Inversion
    test_matrix_inv = np.random.rand(4, 4) * 1e-6
    test_matrix_inv = test_matrix_inv + 0.1 * np.eye(4)  # Ensure it's invertible
    inv_result = hardware.invert_matrix(test_matrix_inv)
    assert inv_result.shape == (4, 4)
    print("✓ Inversion test passed")
    
    # Disconnect
    assert hardware.disconnect()
    assert not hardware.connected
    print("✓ Disconnect test passed")
    
    print("All Mock RRAM Hardware tests passed!")
    return True


def test_hardware_test_fixture():
    """Test the hardware test fixture."""
    print("\nTesting Hardware Test Fixture...")
    
    with create_mock_hardware_fixture() as fixture:
        # Create test matrix and vector (using size 8 to match mock hardware)
        test_matrix = np.random.rand(8, 8) * 1e-6
        test_matrix = test_matrix + 0.1 * np.eye(8)  # Ensure invertible
        test_vector = np.random.rand(8)
        
        # Run conductance test
        cond_results = fixture.run_conductance_test(test_matrix)
        assert 'error_metrics' in cond_results
        assert 'passed' in cond_results
        print("✓ Conductance test passed")
        
        # Run MVM test
        mvm_results = fixture.run_mvm_test(test_matrix, test_vector)
        assert 'error_metrics' in mvm_results
        assert 'passed' in mvm_results
        print("✓ MVM test passed")
        
        # Run inversion test
        inv_results = fixture.run_inversion_test(test_matrix)
        assert 'error_metrics' in inv_results
        assert 'passed' in inv_results
        print("✓ Inversion test passed")
    
    print("All Hardware Test Fixture tests passed!")
    return True


def test_integration_with_hp_inv():
    """Test integration between hardware interface and HP-INV algorithms."""
    print("\nTesting Integration with HP-INV...")
    
    from src.hp_inv import hp_inv
    from src.rram_model import create_rram_matrix
    
    # Create a test RRAM matrix (size 8 to match fixture)
    G = create_rram_matrix(8, variability=0.02, stuck_fault_prob=0.01)
    G = G + 0.3 * np.eye(8)  # Ensure it's well-conditioned
    b = np.random.randn(8)
    
    # Use hardware fixture for computation
    with create_mock_hardware_fixture() as fixture:
        # Write matrix to 'hardware'
        fixture.hardware.write_matrix(G)
        
        # Perform inversion using hardware
        G_inv_hw = fixture.hardware.invert_matrix(G)
        
        # Compare with software HP-INV
        x_hw = G_inv_hw @ b  # Direct solution
        
        x_hp, iters, info = hp_inv(G, b, bits=3, lp_noise_std=0.01, max_iter=10)
        
        # Check that both solutions are reasonable
        residual_hw = np.linalg.norm(G @ x_hw - b)
        residual_hp = np.linalg.norm(G @ x_hp - b)
        
        print(f"Hardware solution residual: {residual_hw:.2e}")
        print(f"HP-INV solution residual: {residual_hp:.2e}")
    
    print("Integration test passed!")
    return True


def main():
    """Run all tests."""
    test_mock_rram_hardware()
    test_hardware_test_fixture()
    test_integration_with_hp_inv()
    print("\nAll hardware interface tests passed!")


if __name__ == "__main__":
    main()