import numpy as np
from typing import Optional, Tuple
import warnings
import sys
import os
# Add the project root to path for potential optimization modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .utils import validate_matrix_vector_inputs, validate_parameter, quantize, validate_matrix_inputs


def hp_inv(G: np.ndarray, b: np.ndarray, bits: int = 3, lp_noise_std: float = 0.01,
           max_iter: int = 10, tol: float = 1e-6,
           relaxation_factor: float = 1.0,
           precomputed_inv: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, dict]:
    """
    Simulate HP-INV iterative refinement for solving G x = b with advanced convergence options.

    Args:
        G: Conductance matrix (with variability)
        b: Right-hand side vector
        bits: Bit precision for LP-INV quantization
        lp_noise_std: Standard deviation of noise added to LP-INV
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        relaxation_factor: Relaxation factor for convergence acceleration (0 < factor <= 2)
        precomputed_inv: Optional precomputed approximate inverse (A0_inv) to use. 
                         If provided, bits and lp_noise_std are ignored for A0_inv creation.

    Returns:
        Tuple of (solution x, iterations taken, convergence info dict)
    """

    # Validate inputs
    validate_matrix_vector_inputs(G, b, "hp_inv")
    validate_parameter(bits, "bits", min_value=1, integer=True, func_name="hp_inv")
    validate_parameter(lp_noise_std, "lp_noise_std", min_value=0, func_name="hp_inv")
    validate_parameter(max_iter, "max_iter", min_value=1, integer=True, func_name="hp_inv")
    validate_parameter(tol, "tol", min_value=0, func_name="hp_inv")
    validate_parameter(relaxation_factor, "relaxation_factor", min_value=0, max_value=2, func_name="hp_inv")

    if precomputed_inv is not None:
        # Use the provided approximate inverse
        validate_matrix_inputs(precomputed_inv, allow_empty=False, func_name="hp_inv (precomputed_inv)")
        if precomputed_inv.shape != G.shape:
            raise ValueError(f"precomputed_inv shape {precomputed_inv.shape} must match G shape {G.shape}")
        A0_inv = precomputed_inv
    else:
        # LP-INV: Quantize and invert with noise
        G_lp = quantize(G, bits)

        try:
            G_lp_inv = np.linalg.inv(G_lp)
            noise = np.random.normal(0, lp_noise_std, G_lp_inv.shape)
            A0_inv = G_lp_inv + noise
        except np.linalg.LinAlgError:
            # Singular matrix, return zero correction
            A0_inv = np.zeros_like(G_lp)
            warnings.warn("Matrix is singular, using zero inverse approximation")

    x = np.zeros_like(b, dtype=float)
    residuals = []

    for k in range(max_iter):
        # HP-MVM: Compute residual r = b - G x
        Ax = G @ x
        r = b - Ax
        residual_norm = np.linalg.norm(r)
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
        info['condition_number_estimate'] = np.linalg.cond(G)
    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(f"Failed to compute condition number: {e}")
        info['condition_number_estimate'] = float('inf')
    
    return x, k + 1, info


def block_hp_inv(G: np.ndarray, b: np.ndarray, block_size: int = 4, **kwargs) -> Tuple[np.ndarray, int, dict]:
    """
    Block HP-INV implementation for large matrices using BlockAMC algorithm.
    
    This implementation uses the BlockAMC algorithm (recursive block inversion) to compute 
    a high-quality approximate inverse, which is then used as the preconditioner 
    in the HP-INV iterative refinement loop.

    Args:
        G: Large conductance matrix to invert
        b: Right-hand side vector
        block_size: Size of submatrix blocks (physical RRAM tile size)
        **kwargs: Additional arguments passed to hp_inv

    Returns:
        Tuple of (solution x, iterations taken, convergence info)

    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If matrix/vector dimensions are invalid
    """
    # Validate inputs
    validate_matrix_vector_inputs(G, b, "block_hp_inv")
    validate_parameter(block_size, "block_size", min_value=1, integer=True, func_name="block_hp_inv")

    n = G.shape[0]
    if n <= block_size:
        # Matrix is small enough, use standard HP-INV
        return hp_inv(G, b, **kwargs)
    
    # 1. Quantize G to simulate the hardware low-precision view
    bits = kwargs.get('bits', 3)
    G_lp = quantize(G, bits)
    
    # 2. Compute the approximate inverse using the BlockAMC (recursive) method
    # This handles large matrices by breaking them down recursively
    try:
        A0_inv = recursive_block_inversion(G_lp, block_size=block_size)
        
        # Add noise to the inverse if requested (simulating analog noise in the inversion step)
        lp_noise_std = kwargs.get('lp_noise_std', 0.01)
        if lp_noise_std > 0:
            noise = np.random.normal(0, lp_noise_std, A0_inv.shape)
            A0_inv += noise
            
    except np.linalg.LinAlgError:
        warnings.warn("Block inversion failed (singular), falling back to zero inverse.")
        A0_inv = np.zeros_like(G)

    # 3. Run the standard HP-INV refinement using this block-derived preconditioner
    # We pass 'precomputed_inv' to skip the internal inversion in hp_inv
    return hp_inv(G, b, precomputed_inv=A0_inv, **kwargs)


def blockamc_inversion(G: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Implement the full BlockAMC algorithm for matrix inversion.
    
    This acts as a wrapper around the robust recursive_block_inversion implementation.
    
    Args:
        G: Large matrix to invert
        block_size: Size of physical RRAM tiles (default 8 for 8x8 arrays)

    Returns:
        Inverted matrix

    Raises:
        TypeError: If inputs are not correct types
        ValueError: If matrix dimensions or parameters are invalid
    """
    # Validate inputs
    validate_matrix_inputs(G, allow_empty=False, func_name="blockamc_inversion")
    validate_parameter(block_size, "block_size", min_value=1, integer=True, func_name="blockamc_inversion")

    return recursive_block_inversion(G, block_size=block_size)


def recursive_block_inversion(G: np.ndarray, block_size: int = 8, depth: int = 0, max_depth: int = 3) -> np.ndarray:
    """
    A recursive implementation of block matrix inversion based on BlockAMC principles.
    This function performs recursive block Gaussian elimination.

    Args:
        G: Matrix to invert
        block_size: Base tile size for physical RRAM arrays
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite recursion

    Returns:
        Inverted matrix

    Raises:
        TypeError: If inputs are not correct types (validated on first call only)
        ValueError: If matrix dimensions or parameters are invalid (validated on first call only)
    """
    # Validate only on first call to avoid overhead in recursion
    if depth == 0:
        # Validate inputs
        validate_matrix_inputs(G, allow_empty=False, func_name="recursive_block_inversion")
        validate_parameter(block_size, "block_size", min_value=1, integer=True, func_name="recursive_block_inversion")
        validate_parameter(depth, "depth", min_value=0, integer=True, func_name="recursive_block_inversion")
        validate_parameter(max_depth, "max_depth", min_value=0, integer=True, func_name="recursive_block_inversion")

    n = G.shape[0]
    
    # Base case: if matrix fits in a single block, return direct inverse
    if n <= block_size or depth >= max_depth:
        try:
            return np.linalg.inv(G)
        except np.linalg.LinAlgError:
            # If not invertible, return pseudoinverse
            return np.linalg.pinv(G)
    
    # Divide matrix into 4 blocks (if possible)
    half_n = n // 2
    A = G[:half_n, :half_n]  # Top-left
    B = G[:half_n, half_n:]  # Top-right
    C = G[half_n:, :half_n]  # Bottom-left
    D = G[half_n:, half_n:]  # Bottom-right
    
    # Recursive block inversion formula: inverse of [[A, B], [C, D]] is:
    # [[(A - B*D^(-1)*C)^(-1), -A^(-1)*B*(D - C*A^(-1)*B)^(-1)],
    #  [-(D - C*A^(-1)*B)^(-1)*C*A^(-1), (D - C*A^(-1)*B)^(-1)]]
    
    # Invert D recursively
    D_inv = recursive_block_inversion(D, block_size, depth + 1, max_depth)
    
    # Calculate Schur complement: S = A - B*D^(-1)*C
    S = A - B @ D_inv @ C
    
    # Invert S recursively
    S_inv = recursive_block_inversion(S, block_size, depth + 1, max_depth)
    
    # Calculate the other Schur complement: T = D - C*A^(-1)*B
    # But first we need A_inv, so calculate that too
    A_inv = recursive_block_inversion(A, block_size, depth + 1, max_depth)
    T = D - C @ A_inv @ B
    T_inv = recursive_block_inversion(T, block_size, depth + 1, max_depth)
    
    # Now construct the inverse matrix
    inv = np.zeros_like(G)
    
    # Top-left: S^(-1)
    inv[:half_n, :half_n] = S_inv
    
    # Top-right: -S^(-1)*B*D^(-1)
    inv[:half_n, half_n:] = -S_inv @ B @ D_inv
    
    # Bottom-left: -D^(-1)*C*S^(-1)
    inv[half_n:, :half_n] = -D_inv @ C @ S_inv
    
    # Bottom-right: T^(-1)
    inv[half_n:, half_n:] = T_inv
    
    return inv


def adaptive_hp_inv(G: np.ndarray, b: np.ndarray, 
                    initial_bits: int = 3,
                    max_bits: int = 6,
                    **kwargs) -> Tuple[np.ndarray, int, dict]:
    """
    Adaptive HP-INV that adjusts quantization precision based on convergence.
    
    Args:
        G: Conductance matrix
        b: Right-hand side vector
        initial_bits: Initial quantization bits
        max_bits: Maximum allowed bits
        **kwargs: Additional arguments passed to hp_inv
        
    Returns:
        Tuple of (solution x, iterations taken, convergence info)
    """
    bits = initial_bits
    x = np.zeros_like(b, dtype=float)
    total_iterations = 0
    all_residuals = []
    
    max_iter = kwargs.get('max_iter', 10)
    tol = kwargs.get('tol', 1e-6)
    
    for outer_iter in range(int(max_iter / 2)):  # Allow up to half the iterations for bit adjustments
        # Solve with current precision
        x_new, inner_iters, info = hp_inv(G, b, bits=bits, **{k: v for k, v in kwargs.items() if k not in ['bits', 'max_iter']})
        
        # Update solution
        x = x_new
        all_residuals.extend(info['residuals'])
        
        # Check if we're converging slowly and need higher precision
        if len(info['residuals']) == max_iter and info['final_residual'] > tol and bits < max_bits:
            bits += 1  # Increase precision
        elif info['converged']:
            total_iterations += inner_iters
            break
            
        total_iterations += inner_iters
        
        if info['final_residual'] < tol:
            break
    
    # Ensure at least one iteration is counted
    if total_iterations == 0 and len(all_residuals) > 0:
        total_iterations = 1
    
    info = {
        'residuals': all_residuals,
        'final_residual': all_residuals[-1] if all_residuals else 0,
        'converged': all_residuals[-1] < tol if all_residuals else False,
        'final_bits': bits,
        'total_iterations': total_iterations
    }
    
    return x, total_iterations, info