import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
# Remove pytest import to run without it
from src.hp_inv import hp_inv

def test_hp_inv_simple():
    """Test HP-INV on a simple well-conditioned matrix."""
    n = 4
    np.random.seed(42)
    A = np.random.randn(n, n) + 5 * np.eye(n)  # Make well-conditioned
    b = np.random.randn(n)
    x_approx, iters, info = hp_inv(A, b, max_iter=20)
    x_true = np.linalg.solve(A, b)
    rel_error = np.linalg.norm(x_approx - x_true) / np.linalg.norm(x_true)
    if not (rel_error < 1e-2):  # Allow some error due to LP noise
        raise AssertionError(f"Relative error {rel_error} is too high (should be < 1e-2)")

def test_hp_inv_identity():
    """Test HP-INV on identity matrix (should converge quickly)."""
    n = 4
    A = np.eye(n)
    b = np.ones(n)
    x_approx, iters, info = hp_inv(A, b)
    if not np.allclose(x_approx, b, atol=1e-3):
        raise AssertionError("Solution should match b for identity matrix")
    if not (iters <= 5):  # Should converge fast
        raise AssertionError(f"Should converge in <= 5 iterations, got {iters}")

if __name__ == "__main__":
    test_hp_inv_simple()
    test_hp_inv_identity()
    print("All HP-INV tests passed!")