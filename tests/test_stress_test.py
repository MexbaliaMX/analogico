import numpy as np
from src.stress_test import run_stress_test

def test_run_stress_test():
    """Test the stress test function."""
    iters, errors = run_stress_test(num_samples=5, n=4)
    assert len(iters) == 5
    assert len(errors) == 5
    assert all(i > 0 for i in iters)
    assert all(e >= 0 for e in errors)