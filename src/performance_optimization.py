"""
Performance Optimization Module for Analogico

This module implements performance optimization enhancements including:
- Multi-core CPU acceleration
- GPU acceleration using JAX
- Algorithm optimizations
- Sparse matrix support
- Mixed-precision computing
"""
import numpy as np
import multiprocessing as mp
from typing import Tuple, Optional, Callable, Any
import time
import warnings
from functools import partial
import sys
import os

# Try to import performance optimization libraries
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap, grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available, GPU acceleration disabled")

try:
    from numba import jit as numba_jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available, CPU JIT optimization disabled")


class PerformanceOptimizer:
    """
    Main class for performance optimization of RRAM algorithms.
    """
    
    def __init__(self, use_jax: bool = True, use_numba: bool = True, n_cores: Optional[int] = None):
        """
        Initialize the performance optimizer.
        
        Args:
            use_jax: Whether to use JAX for GPU acceleration
            use_numba: Whether to use Numba for CPU JIT compilation
            n_cores: Number of CPU cores to use (default: all available)
        """
        self.use_jax = use_jax and JAX_AVAILABLE
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.n_cores = n_cores or mp.cpu_count()
        
        # Set JAX to use GPU if available
        if self.use_jax:
            try:
                jax.config.update('jax_enable_x64', True)  # Enable 64-bit precision
            except:
                pass  # May already be set
    
    def optimize_rram_model(self):
        """Optimize RRAM model functions for performance."""
        if self.use_jax:
            return self._optimize_rram_model_jax()
        elif self.use_numba:
            return self._optimize_rram_model_numba()
        else:
            return self._optimize_rram_model_basic()
    
    def _optimize_rram_model_jax(self):
        """JAX-optimized RRAM model functions."""
        @jit
        def quantize_jax(matrix: jnp.ndarray, bits: int) -> jnp.ndarray:
            """JAX-optimized quantization."""
            if matrix.size == 0 or jnp.all(matrix == 0):
                return matrix
            max_val = jnp.max(jnp.abs(matrix))
            levels = 2**bits - 1
            scale = levels / max_val
            quantized = jnp.round(matrix * scale) / scale
            return quantized
        
        @jit
        def mvm_jax(G: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            """JAX-optimized matrix-vector multiplication."""
            return G @ x
        
        return quantize_jax, mvm_jax
    
    def _optimize_rram_model_numba(self):
        """Numba-optimized RRAM model functions."""
        @numba_jit(nopython=True, cache=True)
        def quantize_numba(matrix, bits):
            """Numba-optimized quantization."""
            if matrix.size == 0 or np.all(matrix == 0):
                return matrix.copy()
            
            max_val = np.max(np.abs(matrix))
            if max_val == 0:
                return matrix.copy()
                
            levels = 2**bits - 1
            scale = levels / max_val
            quantized = np.round(matrix * scale) / scale
            return quantized
        
        @numba_jit(nopython=True, cache=True)
        def mvm_numba(G, x):
            """Numba-optimized matrix-vector multiplication."""
            return np.dot(G, x)
        
        return quantize_numba, mvm_numba
    
    def _optimize_rram_model_basic(self):
        """Basic optimized functions."""
        def quantize_basic(matrix, bits):
            """Optimized quantization without JIT."""
            if matrix.size == 0 or np.all(matrix == 0):
                return matrix
            max_val = np.max(np.abs(matrix))
            if max_val == 0:
                return matrix
            levels = 2**bits - 1
            scale = levels / max_val
            quantized = np.round(matrix * scale) / scale
            return quantized
        
        def mvm_basic(G, x):
            """Basic matrix-vector multiplication."""
            return G @ x
        
        return quantize_basic, mvm_basic


def _stress_test_worker(args):
    """Worker function for parallel stress testing."""
    n, seed_offset, variability, stuck_prob, bits, lp_noise_std, max_iter = args
    
    import sys
    import os
    # Add the project root to path to import the modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    import numpy as np
    # Import inside function to avoid circular imports
    from src.rram_model import create_rram_matrix
    from src.hp_inv import hp_inv
    
    np.random.seed(seed_offset)  # Set different seed for each process
    
    try:
        # Generate a single sample using RRAM model
        G = create_rram_matrix(n, variability=variability, stuck_fault_prob=stuck_prob)
        b = np.random.randn(n)

        # Compute true solution
        try:
            x_true = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            return None, None  # Skip singular matrices

        x_approx, iters, _ = hp_inv(G, b, bits=bits, lp_noise_std=lp_noise_std, max_iter=max_iter)

        # Compute relative error
        rel_error = np.linalg.norm(x_approx - x_true) / np.linalg.norm(x_true)
        return iters, rel_error
    except Exception as e:
        return None, None


def parallel_stress_test(n: int, num_samples: int, variability: float = 0.05,
                        stuck_prob: float = 0.01, bits: int = 3, lp_noise_std: float = 0.01,
                        max_iter: int = 10, n_processes: Optional[int] = None) -> Tuple[list, list]:
    """
    Parallel stress test using multiprocessing for better performance.
    
    Args:
        n: Matrix size
        num_samples: Number of Monte Carlo samples
        variability: Conductance variability
        stuck_prob: Stuck-at fault probability
        bits: LP-INV bit precision
        lp_noise_std: LP-INV noise std
        max_iter: Max iterations for HP-INV
        n_processes: Number of processes to use (default: all available)
    
    Returns:
        Tuple of (convergence_iterations, final_relative_errors)
    """
    import sys
    import os
    # Add the project root to path to import the modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    # Determine number of processes
    n_proc = n_processes or mp.cpu_count()
    
    # Prepare arguments for each worker
    worker_args = [(n, i, variability, stuck_prob, bits, lp_noise_std, max_iter) 
                   for i in range(num_samples)]
    
    # Execute in parallel
    with mp.Pool(processes=n_proc) as pool:
        results = pool.map(_stress_test_worker, worker_args)
    
    # Filter and collect results
    convergence_iters = []
    final_errors = []
    
    for iters, error in results:
        if iters is not None and error is not None:
            convergence_iters.append(iters)
            final_errors.append(error)
    
    return convergence_iters, final_errors


def jax_hp_inv(G, b, bits: int = 3, lp_noise_std: float = 0.01,
               max_iter: int = 10, tol: float = 1e-6, 
               relaxation_factor: float = 1.0):
    """
    JAX-optimized HP-INV algorithm for GPU acceleration.
    
    Args:
        G: Conductance matrix (with variability)
        b: Right-hand side vector
        bits: Bit precision for LP-INV quantization
        lp_noise_std: Standard deviation of noise added to LP-INV
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        relaxation_factor: Relaxation factor for convergence acceleration
    
    Returns:
        Tuple of (solution x, iterations taken, convergence info)
    """
    if not JAX_AVAILABLE:
        warnings.warn("JAX not available, falling back to CPU implementation")
        from src.hp_inv import hp_inv
        return hp_inv(G, b, bits, lp_noise_std, max_iter, tol, relaxation_factor)
    
    @jit
    def quantize_jax(matrix, bits):
        """JAX quantization function."""
        if matrix.size == 0 or jnp.all(matrix == 0):
            return matrix
        max_val = jnp.max(jnp.abs(matrix))
        if max_val == 0:
            return matrix
        levels = 2**bits - 1
        scale = levels / max_val
        quantized = jnp.round(matrix * scale) / scale
        return quantized

    # Convert to JAX arrays
    G_jax = jnp.array(G)
    b_jax = jnp.array(b)
    
    # LP-INV: Quantize and invert with noise
    G_lp = quantize_jax(G_jax, bits)
    
    try:
        G_lp_inv = jnp.linalg.inv(G_lp)
        noise = jax.random.normal(jax.random.PRNGKey(42), G_lp_inv.shape) * lp_noise_std
        A0_inv = G_lp_inv + noise
    except:
        # Singular matrix, return zero correction
        A0_inv = jnp.zeros_like(G_lp)

    def body_fun(carry):
        """Body function for the iterative loop."""
        x, k, residuals = carry
        
        # HP-MVM: Compute residual r = b - G x
        Ax = G_jax @ x
        r = b_jax - Ax
        residual_norm = jnp.linalg.norm(r)
        residuals = residuals.at[k].set(residual_norm)
        
        # Update x
        delta = A0_inv @ r
        x_new = x + relaxation_factor * delta
        
        return x_new, k + 1, residuals

    def cond_fun(carry):
        """Condition function for the loop."""
        x, k, residuals = carry
        return (k < max_iter) & (residuals[k-1] > tol if k > 0 else True)

    # Initialize
    x = jnp.zeros_like(b_jax, dtype=jnp.float64)
    residuals = jnp.full(max_iter, jnp.inf)
    
    # Run the iterative loop
    x_final, k_final, residuals_final = jax.lax.while_loop(
        cond_fun, body_fun, (x, 0, residuals)
    )
    
    # Convert back to numpy
    x_result = np.array(x_final)
    residuals_result = np.array(residuals_final)
    
    info = {
        'residuals': residuals_result[:k_final+1].tolist(),
        'final_residual': float(residuals_result[k_final-1]) if k_final > 0 else float(residuals_result[0]),
        'converged': residuals_result[k_final-1] < tol if k_final > 0 else False,
        'condition_number_estimate': float(np.linalg.cond(G)) if k_final > 0 else float('inf')
    }
    
    return x_result, int(k_final), info


class SparseRRAMModel:
    """
    Sparse matrix implementation for RRAM models to improve performance
    on large systems with sparse conductance patterns.
    """
    
    def __init__(self):
        try:
            from scipy.sparse import csr_matrix, csc_matrix
            self.csr_matrix = csr_matrix
            self.csc_matrix = csc_matrix
            self.sparse_available = True
        except ImportError:
            self.sparse_available = False
            warnings.warn("Scipy sparse not available, sparse matrices disabled")
    
    def create_sparse_rram_matrix(self, n: int, sparsity: float = 0.1,
                                  conductance_range: Tuple[float, float] = (1e-6, 1e-3),
                                  variability: float = 0.05) -> Optional[np.ndarray]:
        """
        Create a sparse RRAM conductance matrix.
        
        Args:
            n: Matrix size
            sparsity: Fraction of non-zero elements (0.0 to 1.0)
            conductance_range: Range of conductance values (min, max)
            variability: Conductance variability
        
        Returns:
            Sparse conductance matrix or None if scipy is not available
        """
        if not self.sparse_available:
            return None
        
        import scipy.sparse as sp
        
        # Calculate number of non-zero elements
        nnz = int(n * n * sparsity)
        
        # Generate random positions
        rows = np.random.randint(0, n, nnz)
        cols = np.random.randint(0, n, nnz)
        data = np.random.uniform(conductance_range[0], conductance_range[1], nnz)
        
        # Add variability
        data *= np.random.normal(1.0, variability, nnz)
        
        # Create sparse matrix
        G_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        
        # Ensure diagonal dominance to prevent singular matrices
        diag_elements = np.random.uniform(conductance_range[1]*0.5, conductance_range[1], n)
        G_sparse = G_sparse + sp.diags(diag_elements, format='csr')
        
        return G_sparse.toarray()  # Convert to dense for compatibility with existing code


def benchmark_performance():
    """Benchmark the performance of optimized vs unoptimized implementations."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from src.rram_model import create_rram_matrix
    from src.hp_inv import hp_inv
    
    print("Performance Benchmarking")
    print("=" * 40)
    
    # Create test matrices of different sizes
    sizes = [64, 128]  # Smaller sizes for demonstration
    
    for n in sizes:
        print(f"\nTesting with {n}x{n} matrices:")
        
        # Standard implementation timing
        G_std = create_rram_matrix(n, variability=0.03, stuck_fault_prob=0.01)
        G_std = G_std + 0.1 * np.eye(n)  # Ensure conditioning
        b_std = np.random.randn(n)
        
        start_time = time.time()
        x_std, iters_std, info_std = hp_inv(G_std, b_std, max_iter=10)
        std_time = time.time() - start_time
        
        print(f"  Standard HP-INV: {std_time:.4f}s for {n}x{n} matrix")
        
        # JAX implementation timing (if available)
        if JAX_AVAILABLE:
            start_time = time.time()
            x_jax, iters_jax, info_jax = jax_hp_inv(G_std, b_std, max_iter=10)
            jax_time = time.time() - start_time
            
            print(f"  JAX HP-INV: {jax_time:.4f}s for {n}x{n} matrix")
            print(f"  Speedup: {std_time/jax_time:.2f}x")
        
        # Parallel stress test timing
        print(f"  Running parallel stress test with {min(10, n//2)} samples...")
        start_time = time.time()
        iters_parallel, errors_parallel = parallel_stress_test(
            n=min(8, n//8), num_samples=min(10, n//2), n_processes=4
        )
        parallel_time = time.time() - start_time
        
        print(f"  Parallel stress test: {parallel_time:.4f}s for {min(10, n//2)} samples")


class MixedPrecisionOptimizer:
    """
    Mixed-precision computing optimizer for performance improvement
    while maintaining accuracy.
    """
    
    def __init__(self, compute_precision: str = 'float32', result_precision: str = 'float64'):
        """
        Initialize mixed precision optimizer.
        
        Args:
            compute_precision: Precision for internal computations ('float32' or 'float64')
            result_precision: Precision for final results ('float32' or 'float64')
        """
        self.compute_precision = compute_precision
        self.result_precision = result_precision
        
        # Map string to numpy dtype
        self.dtype_map = {
            'float32': np.float32,
            'float64': np.float64
        }
        
        self.compute_dtype = self.dtype_map[compute_precision]
        self.result_dtype = self.dtype_map[result_precision]
    
    def hp_inv_mixed_precision(self, G: np.ndarray, b: np.ndarray, 
                              bits: int = 3, lp_noise_std: float = 0.01,
                              max_iter: int = 10, tol: float = 1e-6) -> Tuple[np.ndarray, int, dict]:
        """
        HP-INV algorithm using mixed precision for performance.
        """
        # Cast to compute precision
        G_comp = G.astype(self.compute_dtype)
        b_comp = b.astype(self.compute_dtype)
        
        def quantize(matrix, bits):
            """Quantize matrix to given bit precision."""
            if matrix.size == 0 or np.all(matrix == 0):
                return matrix
            max_val = np.max(np.abs(matrix))
            if max_val == 0:
                return matrix
            levels = 2**bits - 1
            scale = levels / max_val
            quantized = np.round(matrix * scale) / scale
            return quantized

        # LP-INV: Quantize and invert with noise
        G_lp = quantize(G_comp, bits)
        
        try:
            G_lp_inv = np.linalg.inv(G_lp)
            noise = np.random.normal(0, lp_noise_std, G_lp_inv.shape).astype(self.compute_dtype)
            A0_inv = G_lp_inv + noise
        except np.linalg.LinAlgError:
            # Singular matrix, return zero correction
            A0_inv = np.zeros_like(G_lp, dtype=self.compute_dtype)

        x = np.zeros_like(b_comp, dtype=self.compute_dtype)
        residuals = []
        
        for k in range(max_iter):
            # HP-MVM: Compute residual r = b - G x
            Ax = G_comp @ x
            r = b_comp - Ax
            residual_norm = np.linalg.norm(r)
            residuals.append(residual_norm)
            
            # Update x
            delta = A0_inv @ r
            x += delta
            
            # Check convergence
            if residual_norm < tol:
                break

        # Cast result back to result precision
        x_result = x.astype(self.result_dtype)
        
        info = {
            'residuals': [float(r) for r in residuals],
            'final_residual': float(residuals[-1]) if residuals else 0,
            'converged': residuals[-1] < tol if residuals else False,
            'condition_number_estimate': float(np.linalg.cond(G_comp))
        }
        
        return x_result, k + 1, info


if __name__ == "__main__":
    # Test the performance optimization features
    print("Testing Performance Optimization Features")
    print("=" * 50)
    
    # Test optimizer initialization
    optimizer = PerformanceOptimizer()
    print(f"JAX available: {JAX_AVAILABLE}")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"Using {optimizer.n_cores} CPU cores")
    
    # Test sparse model
    sparse_model = SparseRRAMModel()
    if sparse_model.sparse_available:
        sparse_G = sparse_model.create_sparse_rram_matrix(16, sparsity=0.2)
        print(f"Sparse matrix created: shape {sparse_G.shape if sparse_G is not None else None}")
    
    # Test mixed precision
    mixed_opt = MixedPrecisionOptimizer()
    print("Mixed precision optimizer initialized")
    
    # Run basic performance test
    benchmark_performance()
    print("\nPerformance optimization tests completed!")