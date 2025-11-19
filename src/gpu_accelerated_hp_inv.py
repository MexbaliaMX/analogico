"""
GPU-accelerated algorithms for HP-INV using JAX.

This module provides GPU implementations of the HP-INV algorithms
to accelerate computations on the host side when connected to
RRAM hardware.
"""
import numpy as onp  # Use original numpy for fallback
import warnings
from typing import Optional, Tuple, Dict, Any, Callable

# Defer JAX import to function level to avoid AVX issues during import
JAX_AVAILABLE = None  # Will be set when import is attempted
jnp = onp  # Default to numpy


def gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    global JAX_AVAILABLE
    if JAX_AVAILABLE is None:
        try:
            import jax
            JAX_AVAILABLE = True
        except (ImportError, RuntimeError):
            JAX_AVAILABLE = False
    return JAX_AVAILABLE


def get_array_module(use_gpu: bool = True):
    """
    Get the appropriate array module (numpy or jax.numpy) based on availability.

    Args:
        use_gpu: Whether to try to use GPU acceleration

    Returns:
        Array module (numpy or jax.numpy)
    """
    global JAX_AVAILABLE, jnp
    if JAX_AVAILABLE is None:
        try:
            import jax
            import jax.numpy as jnp_local
            JAX_AVAILABLE = True
            jnp = jnp_local
        except (ImportError, RuntimeError):
            JAX_AVAILABLE = False
            jnp = onp  # Fallback to numpy

    if use_gpu and JAX_AVAILABLE:
        return jnp
    else:
        return onp


def gpu_hp_inv(
    G: onp.ndarray,
    b: onp.ndarray,
    bits: int = 3,
    lp_noise_std: float = 0.01,
    max_iter: int = 10,
    tol: float = 1e-6,
    relaxation_factor: float = 1.0,
    use_gpu: bool = True
) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
    """
    GPU-accelerated HP-INV iterative refinement for solving G x = b.

    Args:
        G: Conductance matrix (with variability)
        b: Right-hand side vector
        bits: Bit precision for LP-INV quantization
        lp_noise_std: Standard deviation of noise added to LP-INV
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        relaxation_factor: Relaxation factor for convergence acceleration
        use_gpu: Whether to use GPU acceleration if available

    Returns:
        Tuple of (solution x, iterations taken, convergence info dict)
    """
    # Check JAX availability
    global JAX_AVAILABLE
    if JAX_AVAILABLE is None:
        try:
            import jax
            import jax.numpy as jnp_local
            from jax import jit as jit_local
            from jax.scipy.linalg import inv as jax_inv_local
            JAX_AVAILABLE = True
        except (ImportError, RuntimeError):
            JAX_AVAILABLE = False

    if not JAX_AVAILABLE or not use_gpu:
        # Fall back to CPU implementation
        return _cpu_hp_inv(G, b, bits, lp_noise_std, max_iter, tol, relaxation_factor)

    try:
        import jax
        import jax.numpy as jnp_local
        from jax import jit
        from jax.scipy.linalg import inv as jax_inv
    except (ImportError, RuntimeError):
        # If JAX import fails at runtime, fall back to CPU
        return _cpu_hp_inv(G, b, bits, lp_noise_std, max_iter, tol, relaxation_factor)

    # Convert to JAX arrays
    G_jax = jnp_local.array(G)
    b_jax = jnp_local.array(b)

    def quantize(matrix, bits):
        """Quantize matrix to given bit precision."""
        if matrix.size == 0 or jnp_local.all(matrix == 0):
            return matrix
        max_val = jnp_local.max(jnp_local.abs(matrix))
        levels = 2**bits - 1
        if max_val == 0:
            return matrix
        scale = levels / max_val
        quantized = jnp_local.round(matrix * scale) / scale
        return quantized

    # LP-INV: Quantize and invert with noise
    G_lp = quantize(G_jax, bits)

    try:
        G_lp_inv = jax_inv(G_lp)
        noise = jax.random.normal(jax.random.PRNGKey(42), G_lp_inv.shape) * lp_noise_std
        A0_inv = G_lp_inv + noise
    except Exception:
        # Singular matrix, return zero correction
        A0_inv = jnp_local.zeros_like(G_lp)
        warnings.warn("Matrix is singular, using zero inverse approximation")

    # Initialize solution
    x = jnp_local.zeros_like(b_jax, dtype=float)
    residuals = []

    # JIT-compiled iterative refinement loop
    @jit
    def refine_step(x, A0_inv, G, b, relaxation_factor):
        # Compute residual r = b - G x
        Ax = G @ x
        r = b - Ax
        residual_norm = jnp_local.linalg.norm(r)

        # Update x with relaxation factor
        delta = A0_inv @ r
        x_new = x + relaxation_factor * delta

        return x_new, residual_norm

    # Run iterative refinement
    for k in range(max_iter):
        x, residual_norm = refine_step(x, A0_inv, G_jax, b_jax, relaxation_factor)
        residuals.append(float(residual_norm))

        # Check convergence
        if residual_norm < tol:
            break

    # Convert back to numpy arrays
    x_result = onp.array(x)
    residuals = [float(r) for r in residuals]

    info = {
        'residuals': residuals,
        'final_residual': residuals[-1] if residuals else 0,
        'converged': residuals[-1] < tol if residuals else False,
        'condition_number_estimate': None
    }

    # Calculate condition number if matrix is invertible
    try:
        info['condition_number_estimate'] = onp.linalg.cond(G)
    except:
        info['condition_number_estimate'] = float('inf')

    return x_result, len(residuals), info


def _cpu_hp_inv(
    G: onp.ndarray, 
    b: onp.ndarray, 
    bits: int, 
    lp_noise_std: float,
    max_iter: int, 
    tol: float,
    relaxation_factor: float
) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
    """
    CPU-only implementation of HP-INV for fallback when JAX is not available.
    """
    def quantize(matrix, bits):
        """Quantize matrix to given bit precision."""
        if matrix.size == 0 or onp.all(matrix == 0):
            return matrix
        max_val = onp.max(onp.abs(matrix))
        levels = 2**bits - 1
        if max_val == 0:
            return matrix
        scale = levels / max_val
        quantized = onp.round(matrix * scale) / scale
        return quantized

    # LP-INV: Quantize and invert with noise
    G_lp = quantize(G, bits)

    try:
        G_lp_inv = onp.linalg.inv(G_lp)
        noise = onp.random.normal(0, lp_noise_std, G_lp_inv.shape)
        A0_inv = G_lp_inv + noise
    except onp.linalg.LinAlgError:
        # Singular matrix, return zero correction
        A0_inv = onp.zeros_like(G_lp)
        warnings.warn("Matrix is singular, using zero inverse approximation")

    x = onp.zeros_like(b, dtype=float)
    residuals = []

    for k in range(max_iter):
        # Compute residual r = b - G x
        Ax = G @ x
        r = b - Ax
        residual_norm = onp.linalg.norm(r)
        residuals.append(residual_norm)

        # Update x with relaxation factor
        delta = A0_inv @ r
        x += relaxation_factor * delta

        # Check convergence
        if residual_norm < tol:
            break

    info = {
        'residuals': residuals,
        'final_residual': residuals[-1] if residuals else 0,
        'converged': residuals[-1] < tol if residuals else False,
        'condition_number_estimate': None
    }

    # Calculate condition number if matrix is invertible
    try:
        info['condition_number_estimate'] = onp.linalg.cond(G)
    except:
        info['condition_number_estimate'] = float('inf')

    return x, k + 1, info


def gpu_block_hp_inv(
    G: onp.ndarray,
    b: onp.ndarray,
    block_size: int = 8,
    use_gpu: bool = True,
    **kwargs
) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
    """
    GPU-accelerated Block HP-INV implementation for large matrices using BlockAMC algorithm.

    Args:
        G: Large conductance matrix to invert
        b: Right-hand side vector
        block_size: Size of submatrix blocks (physical RRAM tile size)
        use_gpu: Whether to use GPU acceleration if available
        **kwargs: Additional arguments passed to hp_inv

    Returns:
        Tuple of (solution x, iterations taken, convergence info)
    """
    # Check JAX availability
    global JAX_AVAILABLE
    if JAX_AVAILABLE is None:
        try:
            import jax
            JAX_AVAILABLE = True
        except (ImportError, RuntimeError):
            JAX_AVAILABLE = False

    if not JAX_AVAILABLE or not use_gpu:
        # Fall back to CPU implementation
        return _cpu_block_hp_inv(G, b, block_size, **kwargs)

    try:
        import jax
        import jax.numpy as jnp_local
        from jax import jit
        from jax.scipy.linalg import inv as jax_inv
    except (ImportError, RuntimeError):
        # If JAX import fails at runtime, fall back to CPU
        return _cpu_block_hp_inv(G, b, block_size, **kwargs)

    # Convert to JAX arrays
    G_jax = jnp_local.array(G)
    b_jax = jnp_local.array(b)

    n = G_jax.shape[0]
    if n <= block_size:
        # Matrix is small enough, use standard GPU HP-INV
        return gpu_hp_inv(
            G, b,
            use_gpu=use_gpu,
            **{k: v for k, v in kwargs.items() if k != 'max_iter'}
        )

    # Initialize solution
    x = jnp_local.zeros_like(b_jax, dtype=float)
    residuals = []

    # Get parameters
    max_iter = kwargs.get('max_iter', 10)
    tol = kwargs.get('tol', 1e-6)
    relaxation_factor = kwargs.get('relaxation_factor', 1.0)

    # JIT-compiled block processing function
    @jit
    def process_block(x, G, b, block_idx, block_size, relaxation_factor):
        n = G.shape[0]
        i = block_idx * block_size
        row_end = jnp_local.minimum(i + block_size, n)
        row_slice = slice(i, row_end)

        # Extract the block of G
        G_block = G[row_slice, row_slice]

        # Get the corresponding part of residual
        Ax = G @ x
        r = b - Ax
        r_block = r[row_slice]

        # Solve for this block using HP-INV (simplified)
        try:
            G_block_inv = jax_inv(G_block)
            x_block = G_block_inv @ r_block
        except:
            # Use pseudoinverse if not invertible
            G_block_inv = jnp_local.linalg.pinv(G_block)
            x_block = G_block_inv @ r_block

        # Update the solution for this block
        x_new = x.at[row_slice].add(relaxation_factor * x_block)
        return x_new, jnp_local.linalg.norm(r)

    # Process blocks iteratively
    for k in range(max_iter):
        x_new = x
        residual_norm = 0.0

        # Process all blocks in the matrix
        num_blocks = int(jnp_local.ceil(n / block_size))
        for block_idx in range(num_blocks):
            x_new, residual_norm = process_block(
                x_new, G_jax, b_jax, block_idx, block_size, relaxation_factor
            )

        x = x_new
        residuals.append(float(residual_norm))

        # Check for convergence
        if residual_norm < tol:
            break

    # Convert back to numpy arrays
    x_result = onp.array(x)
    residuals = [float(r) for r in residuals]

    info = {
        'residuals': residuals,
        'final_residual': residuals[-1] if residuals else 0,
        'converged': residuals[-1] < tol if residuals else False,
    }

    return x_result, len(residuals), info


def _cpu_block_hp_inv(
    G: onp.ndarray,
    b: onp.ndarray,
    block_size: int,
    **kwargs
) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
    """
    CPU-only implementation of Block HP-INV for fallback.
    """
    try:
        from .hp_inv import hp_inv  # Import here to avoid circular import
    except ImportError:
        # Handle case where module is imported directly
        import sys
        import os
        # Add the src directory to path if not already there
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from hp_inv import hp_inv

    n = G.shape[0]
    if n <= block_size:
        # Matrix is small enough, use standard HP-INV
        return hp_inv(G, b, **{k: v for k, v in kwargs.items() if k != 'max_iter'})

    # Partition the matrix into blocks
    x = onp.zeros_like(b, dtype=float)
    residuals = []

    # Initialize using block approach
    for k in range(kwargs.get('max_iter', 10)):
        # Compute residual
        r = b - G @ x
        residual_norm = onp.linalg.norm(r)
        residuals.append(residual_norm)

        # Process in blocks to simulate the tile-based approach
        x_new = x.copy()
        for i in range(0, n, block_size):
            end_i = min(i + block_size, n)
            row_indices = slice(i, end_i)

            # Extract the block of G
            G_block = G[row_indices, row_indices]

            # Get the corresponding part of residual
            r_block = r[row_indices]

            # Solve for this block using HP-INV
            if G_block.shape[0] == G_block.shape[1]:  # Make sure it's square
                # Use hp_inv function from the main module
                x_block, _, _ = hp_inv(G_block, r_block, **{k: v for k, v in kwargs.items() if k != 'max_iter'})

                # Update the solution for this block
                x_new[row_indices] += kwargs.get('relaxation_factor', 1.0) * x_block
            else:
                # For non-square blocks (at the end if matrix size not divisible by block_size)
                # We'll use least squares approach
                x_block = onp.linalg.lstsq(G_block, r_block, rcond=None)[0]
                x_new[row_indices] += kwargs.get('relaxation_factor', 1.0) * x_block

        x = x_new

        # Check for convergence
        if residual_norm < kwargs.get('tol', 1e-6):
            break

    info = {
        'residuals': residuals,
        'final_residual': residuals[-1] if residuals else 0,
        'converged': residuals[-1] < kwargs.get('tol', 1e-6) if residuals else False,
    }

    return x, len(residuals), info


def gpu_recursive_block_inversion(
    G: onp.ndarray,
    block_size: int = 8,
    use_gpu: bool = True,
    depth: int = 0,
    max_depth: int = 3
) -> onp.ndarray:
    """
    GPU-accelerated recursive implementation of block matrix inversion based on BlockAMC principles.

    Args:
        G: Matrix to invert
        block_size: Base tile size for physical RRAM arrays
        use_gpu: Whether to use GPU acceleration if available
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite recursion

    Returns:
        Inverted matrix
    """
    # Check JAX availability
    global JAX_AVAILABLE
    if JAX_AVAILABLE is None:
        try:
            import jax
            JAX_AVAILABLE = True
        except (ImportError, RuntimeError):
            JAX_AVAILABLE = False

    if not JAX_AVAILABLE or not use_gpu:
        # Fall back to CPU implementation
        return _cpu_recursive_block_inversion(G, block_size, depth, max_depth)

    try:
        import jax
        import jax.numpy as jnp_local
        from jax.scipy.linalg import inv as jax_inv
    except (ImportError, RuntimeError):
        # If JAX import fails at runtime, fall back to CPU
        return _cpu_recursive_block_inversion(G, block_size, depth, max_depth)

    # Convert to JAX arrays
    G_jax = jnp_local.array(G)
    n = G_jax.shape[0]

    # Base case: if matrix fits in a single block, return direct inverse
    if n <= block_size or depth >= max_depth:
        try:
            result = jax_inv(G_jax)
            return onp.array(result)
        except Exception:
            # If not invertible, return pseudoinverse
            result = jnp_local.linalg.pinv(G_jax)
            return onp.array(result)

    # Divide matrix into 4 blocks (if possible)
    half_n = n // 2
    A = G_jax[:half_n, :half_n]  # Top-left
    B = G_jax[:half_n, half_n:]  # Top-right
    C = G_jax[half_n:, :half_n]  # Bottom-left
    D = G_jax[half_n:, half_n:]  # Bottom-right

    # Recursive block inversion formula

    # Invert D recursively
    D_inv = jnp_local.array(
        gpu_recursive_block_inversion(
            onp.array(D), block_size, use_gpu, depth + 1, max_depth
        )
    )

    # Calculate Schur complement: S = A - B*D^(-1)*C
    S = A - B @ D_inv @ C

    # Invert S recursively
    S_inv = jnp_local.array(
        gpu_recursive_block_inversion(
            onp.array(S), block_size, use_gpu, depth + 1, max_depth
        )
    )

    # Calculate the other Schur complement: T = D - C*A^(-1)*B
    A_inv = jnp_local.array(
        gpu_recursive_block_inversion(
            onp.array(A), block_size, use_gpu, depth + 1, max_depth
        )
    )
    T = D - C @ A_inv @ B
    T_inv = jnp_local.array(
        gpu_recursive_block_inversion(
            onp.array(T), block_size, use_gpu, depth + 1, max_depth
        )
    )

    # Now construct the inverse matrix
    inv = jnp_local.zeros_like(G_jax)

    # Top-left: S^(-1)
    inv = inv.at[:half_n, :half_n].set(S_inv)

    # Top-right: -S^(-1)*B*D^(-1)
    inv = inv.at[:half_n, half_n:].set(-S_inv @ B @ D_inv)

    # Bottom-left: -D^(-1)*C*S^(-1)
    inv = inv.at[half_n:, :half_n].set(-D_inv @ C @ S_inv)

    # Bottom-right: T^(-1)
    inv = inv.at[half_n:, half_n:].set(T_inv)

    return onp.array(inv)


def _cpu_recursive_block_inversion(
    G: onp.ndarray,
    block_size: int,
    depth: int,
    max_depth: int
) -> onp.ndarray:
    """
    CPU-only recursive block inversion for fallback.
    """
    try:
        from .hp_inv import recursive_block_inversion
    except ImportError:
        # Handle case where module is imported directly
        import sys
        import os
        # Add the src directory to path if not already there
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from hp_inv import recursive_block_inversion

    return recursive_block_inversion(G, block_size, depth, max_depth)


class GPUAcceleratedHPINV:
    """
    A wrapper class for GPU-accelerated HP-INV operations.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the GPU-accelerated HP-INV solver.

        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.use_gpu = use_gpu and gpu_available()
        self.xp = get_array_module(self.use_gpu)
        
    def solve(self, G: onp.ndarray, b: onp.ndarray, **kwargs) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
        """
        Solve the linear system G*x = b using GPU-accelerated HP-INV.

        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters for HP-INV

        Returns:
            Tuple of (solution, iterations, info)
        """
        if self.use_gpu:
            return gpu_hp_inv(G, b, use_gpu=True, **kwargs)
        else:
            # Filter kwargs to only include those that standard hp_inv accepts
            valid_hp_inv_kwargs = {k: v for k, v in kwargs.items()
                                   if k in ['bits', 'lp_noise_std', 'max_iter', 'tol', 'relaxation_factor']}

            try:
                from .hp_inv import hp_inv
                return hp_inv(G, b, **valid_hp_inv_kwargs)
            except ImportError:
                # Handle case where module is imported directly
                import sys
                import os
                # Add the src directory to path if not already there
                src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                from hp_inv import hp_inv
                return hp_inv(G, b, **valid_hp_inv_kwargs)
    
    def block_solve(self, G: onp.ndarray, b: onp.ndarray, **kwargs) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
        """
        Solve the linear system using block HP-INV with GPU acceleration.

        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters for block HP-INV

        Returns:
            Tuple of (solution, iterations, info)
        """
        if self.use_gpu:
            return gpu_block_hp_inv(G, b, use_gpu=True, **kwargs)
        else:
            # Filter kwargs to only include those that standard block_hp_inv accepts
            valid_block_hp_inv_kwargs = {k: v for k, v in kwargs.items()
                                         if k in ['bits', 'lp_noise_std', 'max_iter', 'tol', 'relaxation_factor', 'block_size']}

            try:
                from .hp_inv import block_hp_inv
                return block_hp_inv(G, b, **valid_block_hp_inv_kwargs)
            except ImportError:
                # Handle case where module is imported directly
                import sys
                import os
                # Add the src directory to path if not already there
                src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                from hp_inv import block_hp_inv
                return block_hp_inv(G, b, **valid_block_hp_inv_kwargs)
    
    def invert_matrix(self, G: onp.ndarray, **kwargs) -> onp.ndarray:
        """
        Invert a matrix using recursive block inversion with GPU acceleration.

        Args:
            G: Matrix to invert
            **kwargs: Additional parameters

        Returns:
            Inverted matrix
        """
        if self.use_gpu:
            block_size = kwargs.get('block_size', 8)
            return gpu_recursive_block_inversion(G, block_size=block_size, use_gpu=True)
        else:
            from .hp_inv import recursive_block_inversion
            block_size = kwargs.get('block_size', 8)
            return recursive_block_inversion(G, block_size=block_size)


def adaptive_gpu_hp_inv(
    G: onp.ndarray,
    b: onp.ndarray,
    initial_bits: int = 2,
    max_bits: int = 8,
    convergence_threshold: float = 0.1,
    max_iter: int = 20,
    tol: float = 1e-6,
    use_gpu: bool = True,
    **kwargs
) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
    """
    Adaptive HP-INV that adjusts quantization precision based on convergence,
    using GPU acceleration when available.

    Args:
        G: Conductance matrix
        b: Right-hand side vector
        initial_bits: Initial quantization bits
        max_bits: Maximum allowed bits
        convergence_threshold: Threshold to increase precision
        max_iter: Maximum iterations
        tol: Convergence tolerance
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional arguments

    Returns:
        Tuple of (solution x, iterations taken, convergence info)
    """
    # Check JAX availability
    global JAX_AVAILABLE
    if JAX_AVAILABLE is None:
        try:
            import jax
            import jax.numpy as jnp_local
            JAX_AVAILABLE = True
        except (ImportError, RuntimeError):
            JAX_AVAILABLE = False

    if not JAX_AVAILABLE or not use_gpu:
        # Fall back to CPU implementation
        return _cpu_adaptive_hp_inv(G, b, initial_bits, max_bits,
                                  convergence_threshold, max_iter, tol, **kwargs)

    try:
        import jax
        import jax.numpy as jnp_local
    except (ImportError, RuntimeError):
        # If JAX import fails at runtime, fall back to CPU
        return _cpu_adaptive_hp_inv(G, b, initial_bits, max_bits,
                                  convergence_threshold, max_iter, tol, **kwargs)

    bits = initial_bits
    x = jnp_local.array(onp.zeros_like(b, dtype=float))
    total_iterations = 0
    all_residuals = []

    # Get solver function based on availability
    solve_func = gpu_hp_inv if use_gpu and JAX_AVAILABLE else _cpu_hp_inv

    max_inner_iter = kwargs.get('max_inner_iter', 5)  # Max inner iterations per precision level

    for outer_iter in range(int(max_iter / max_inner_iter)):  # Allow up to half the iterations for bit adjustments
        # Solve with current precision using fixed number of iterations
        x_new, inner_iters, info = solve_func(
            G, onp.array(b),
            bits=bits,
            max_iter=max_inner_iter,
            use_gpu=use_gpu,
            **{k: v for k, v in kwargs.items() if k not in ['bits', 'max_iter']}
        )

        # Convert result back to jax array if needed
        x_new_jax = jnp_local.array(x_new)

        # Update solution
        x = x_new_jax
        all_residuals.extend(info['residuals'])

        # Check if we're converging slowly and need higher precision
        if len(info['residuals']) >= 2:  # Need at least 2 residuals to check convergence rate
            recent_improvement = info['residuals'][-2] - info['residuals'][-1]
            if recent_improvement < convergence_threshold and bits < max_bits:
                bits += 1  # Increase precision
                print(f"Increasing precision to {bits} bits due to slow convergence")

        total_iterations += inner_iters

        # Check for overall convergence
        if info['converged'] or (all_residuals and all_residuals[-1] < tol):
            break

        if total_iterations >= max_iter:
            break

    # Convert final result back to numpy
    final_x = onp.array(x)

    info = {
        'residuals': all_residuals,
        'final_residual': all_residuals[-1] if all_residuals else 0,
        'converged': all_residuals[-1] < tol if all_residuals else False,
        'final_bits': bits,
        'total_iterations': total_iterations,
        'bits_used': list(range(initial_bits, bits+1))  # Track the bits used
    }

    return final_x, total_iterations, info


def _cpu_adaptive_hp_inv(
    G: onp.ndarray,
    b: onp.ndarray,
    initial_bits: int,
    max_bits: int,
    convergence_threshold: float,
    max_iter: int,
    tol: float,
    **kwargs
) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
    """
    CPU-only adaptive HP-INV for fallback.
    """
    try:
        from .hp_inv import adaptive_hp_inv
    except ImportError:
        # Handle case where module is imported directly
        import sys
        import os
        # Add the src directory to path if not already there
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from hp_inv import adaptive_hp_inv

    return adaptive_hp_inv(G, b,
                          initial_bits=initial_bits,
                          max_bits=max_bits,
                          **kwargs)


class AdaptivePrecisionHPINV:
    """
    Adaptive precision solver that adjusts quantization levels based on hardware feedback.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the adaptive precision HP-INV solver.

        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.use_gpu = use_gpu and gpu_available()
        self.xp = get_array_module(self.use_gpu)

    def solve_with_hardware_feedback(
        self,
        G: onp.ndarray,
        b: onp.ndarray,
        hardware_interface: Optional[object] = None,
        **kwargs
    ) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
        """
        Solve using adaptive precision with hardware feedback.

        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            hardware_interface: Optional hardware interface for feedback
            **kwargs: Additional parameters

        Returns:
            Tuple of (solution, iterations, info)
        """
        # Collect hardware feedback if available
        if hardware_interface:
            # Get noise characteristics from hardware
            noise_level = getattr(hardware_interface, 'lp_noise_std', 0.01)
            # Adjust the convergence threshold based on hardware noise
            convergence_threshold = kwargs.get('convergence_threshold', 0.1) * (1 + noise_level/0.01)
            kwargs['convergence_threshold'] = convergence_threshold

            # Get temperature effects
            if hasattr(hardware_interface, 'temperature'):
                # Adjust algorithm parameters based on temperature
                temp_factor = 1.0 + abs(hardware_interface.temperature - 300.0) * 0.001
                kwargs['relaxation_factor'] = kwargs.get('relaxation_factor', 1.0) / temp_factor

        # Use adaptive solver with GPU if available
        if self.use_gpu:
            return adaptive_gpu_hp_inv(G, b, use_gpu=True, **kwargs)
        else:
            from .hp_inv import adaptive_hp_inv
            return adaptive_hp_inv(G, b, **kwargs)

    def solve(
        self,
        G: onp.ndarray,
        b: onp.ndarray,
        **kwargs
    ) -> Tuple[onp.ndarray, int, Dict[str, Any]]:
        """
        Solve using adaptive precision without hardware-specific feedback.

        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters

        Returns:
            Tuple of (solution, iterations, info)
        """
        if self.use_gpu:
            return adaptive_gpu_hp_inv(G, b, use_gpu=True, **kwargs)
        else:
            # Filter kwargs to only include those that standard adaptive_hp_inv accepts
            valid_adaptive_kwargs = {k: v for k, v in kwargs.items()
                                     if k in ['initial_bits', 'max_bits', 'max_iter', 'tol']}

            try:
                from .hp_inv import adaptive_hp_inv
                return adaptive_hp_inv(G, b, **valid_adaptive_kwargs)
            except ImportError:
                # Handle case where module is imported directly
                import sys
                import os
                # Add the src directory to path if not already there
                src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                from hp_inv import adaptive_hp_inv
                return adaptive_hp_inv(G, b, **valid_adaptive_kwargs)