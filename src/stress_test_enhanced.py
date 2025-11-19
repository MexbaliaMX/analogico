"""
Enhanced stress test module with performance optimizations.

This module includes both the original stress test and the parallel version.
"""
import numpy as np
from typing import Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .rram_model import create_rram_matrix
from .hp_inv import hp_inv
from .performance_optimization import parallel_stress_test as parallel_stress_test_impl

def run_stress_test(n: int = 8, num_samples: int = 100, variability: float = 0.05,
                    stuck_prob: float = 0.01, bits: int = 3, lp_noise_std: float = 0.01,
                    max_iter: int = 10, temperature: float = 300.0,
                    time_since_programming: float = 0.0, use_advanced_physics: bool = False,
                    material: str = 'HfO2', device_area: float = 1e-12,
                    use_parallel: bool = False, n_processes: int = None) -> Tuple[list, list]:
    """
    Run Monte Carlo stress test for HP-INV under RRAM variability and temperature effects.

    Args:
        n: Matrix size
        num_samples: Number of Monte Carlo samples
        variability: Conductance variability
        stuck_prob: Stuck-at fault probability
        bits: LP-INV bit precision
        lp_noise_std: LP-INV noise std
        max_iter: Max iterations for HP-INV
        temperature: Temperature in Kelvin (default: 300K)
        time_since_programming: Time since programming in seconds (default: 0)
        use_advanced_physics: Whether to use advanced physics-based models (default: False)
        material: RRAM material type ('HfO2', 'TaOx', 'TiO2', etc.) (default: 'HfO2')
        device_area: Physical area of each RRAM device (m²) (default: 1e-12)
        use_parallel: Whether to use parallel processing (default: False)
        n_processes: Number of processes for parallel execution (default: all available)

    Returns:
        Tuple of (convergence_iterations, final_relative_errors)
    """
    if use_parallel:
        # Use parallel implementation
        return parallel_stress_test_impl(
            n=n, 
            num_samples=num_samples, 
            variability=variability, 
            stuck_prob=stuck_prob, 
            bits=bits, 
            lp_noise_std=lp_noise_std,
            max_iter=max_iter,
            n_processes=n_processes
        )
    else:
        # Use original sequential implementation
        convergence_iters = []
        final_errors = []

        for _ in range(num_samples):
            G = create_rram_matrix(n, variability=variability, stuck_fault_prob=stuck_prob,
                                  temperature=temperature, time_since_programming=time_since_programming,
                                  use_advanced_physics=use_advanced_physics, material=material,
                                  device_area=device_area)
            b = np.random.randn(n)

            # Compute true solution
            try:
                x_true = np.linalg.solve(G, b)
            except np.linalg.LinAlgError:
                continue  # Skip singular matrices

            x_approx, iters, _ = hp_inv(G, b, bits=bits, lp_noise_std=lp_noise_std, max_iter=max_iter)

            # Compute relative error
            rel_error = np.linalg.norm(x_approx - x_true) / np.linalg.norm(x_true)
            convergence_iters.append(iters)
            final_errors.append(rel_error)

        return convergence_iters, final_errors


if __name__ == "__main__":
    # Example usage
    print("Running stress test with performance optimizations...")
    
    # Sequential version
    print("Sequential stress test:")
    iters_seq, errors_seq = run_stress_test(n=6, num_samples=10, use_parallel=False)
    print(f"  Completed {len(iters_seq)} samples in sequential mode")
    
    # Parallel version
    print("Parallel stress test:")
    iters_par, errors_par = run_stress_test(n=6, num_samples=10, use_parallel=True)
    print(f"  Completed {len(iters_par)} samples in parallel mode")
    
    print("Stress test enhancement completed successfully!")