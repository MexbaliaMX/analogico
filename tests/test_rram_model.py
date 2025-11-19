import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.rram_model import create_rram_matrix, mvm, DEFAULT_CONDUCTANCE_LEVELS

def test_create_rram_matrix():
    """Test RRAM matrix creation."""
    n = 4
    G = create_rram_matrix(n)
    assert G.shape == (n, n)
    assert np.all(G >= 0)  # Conductance should be non-negative

def test_discrete_conductance_levels():
    """Test that the matrix is created with discrete conductance levels."""
    n = 10
    # Create a matrix with no variability or faults to check the base levels
    G = create_rram_matrix(n, variability=0, stuck_fault_prob=0, line_resistance=0)
    # Flatten the matrix to a 1D array
    g_values = G.flatten()
    # Check that all values in g_values are present in the default conductance levels
    assert np.all(np.isin(g_values, DEFAULT_CONDUCTANCE_LEVELS))

def test_mvm():
    """Test matrix-vector multiplication."""
    n = 4
    G = create_rram_matrix(n)
    x = np.random.randn(n)
    y = mvm(G, x)
    assert y.shape == (n,)
    # Check it's approximately G @ x
    assert np.allclose(y, G @ x)