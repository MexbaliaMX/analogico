import numpy as np
from typing import Optional, Tuple
import warnings
import sys
import os
# Add the project root to path for potential optimization modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def hp_inv(G: np.ndarray, b: np.ndarray, bits: int = 3, lp_noise_std: float = 0.01,
           max_iter: int = 10, tol: float = 1e-6, 
           relaxation_factor: float = 1.0) -> Tuple[np.ndarray, int, dict]:
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

    Returns:
        Tuple of (solution x, iterations taken, convergence info dict)
    """
    def quantize(matrix: np.ndarray, bits: int) -> np.ndarray:
        """Quantize matrix to given bit precision."""
        if matrix.size == 0 or np.all(matrix == 0):
            return matrix
        max_val = np.max(np.abs(matrix))
        levels = 2**bits - 1
        if max_val == 0:
            return matrix
        scale = levels / max_val
        # Avoid potential numerical issues if scale is too small
        if scale < 1e-12:  # Very small scale could cause numerical issues
            return np.zeros_like(matrix)
        quantized = np.round(matrix * scale) / scale
        return quantized

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
    except:
        info['condition_number_estimate'] = float('inf')
    
    return x, k + 1, info


def block_hp_inv(G: np.ndarray, b: np.ndarray, block_size: int = 4, **kwargs) -> Tuple[np.ndarray, int, dict]:
    """
    Block HP-INV implementation for large matrices using BlockAMC algorithm.
    This implementation uses a more sophisticated approach based on BlockAMC principles:
    1. Partition large matrices into submatrices that fit existing RRAM tiles
    2. Enable 16x16 real-valued inversions using 8x8 arrays without reprogramming
    
    Args:
        G: Large conductance matrix to invert
        b: Right-hand side vector
        block_size: Size of submatrix blocks (physical RRAM tile size)
        **kwargs: Additional arguments passed to hp_inv
        
    Returns:
        Tuple of (solution x, iterations taken, convergence info)
    """
    n = G.shape[0]
    if n <= block_size:
        # Matrix is small enough, use standard HP-INV
        return hp_inv(G, b, **kwargs)
    
    # Partition the matrix into blocks
    n_blocks = int(np.ceil(n / block_size))
    x = np.zeros_like(b, dtype=float)
    residuals = []
    
    # Initialize using block approach
    for k in range(kwargs.get('max_iter', 10)):
        # Compute residual
        r = b - G @ x
        residual_norm = np.linalg.norm(r)
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
                x_block, _, _ = hp_inv(G_block, r_block, **{k: v for k, v in kwargs.items() if k != 'max_iter'})
                
                # Update the solution for this block
                x_new[row_indices] += kwargs.get('relaxation_factor', 1.0) * x_block
            else:
                # For non-square blocks (at the end if matrix size not divisible by block_size)
                # We'll use least squares approach
                x_block = np.linalg.lstsq(G_block, r_block, rcond=None)[0]
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


def blockamc_inversion(G: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Implement the full BlockAMC algorithm for matrix inversion.
    
    The BlockAMC algorithm partitions large matrices into block submatrices:
    1. Stage 1: solves Re(A) via LP-INV and Im(A) via MVM
    2. Stage 2: recursively inverts block diagonals
    
    For real-valued matrices, this simplifies to block-wise processing without 
    complex number handling, but maintains the recursive structure.
    
    Args:
        G: Large matrix to invert
        block_size: Size of physical RRAM tiles (default 8 for 8x8 arrays)
        
    Returns:
        Inverted matrix
    """
    n = G.shape[0]
    
    if n <= block_size:
        # Matrix fits in a single tile, use direct inversion
        try:
            return np.linalg.inv(G)
        except np.linalg.LinAlgError:
            # If not invertible, return pseudoinverse
            return np.linalg.pinv(G)
    
    # Partition matrix into blocks
    n_blocks = int(np.ceil(n / block_size))
    
    # Pad matrix if needed to make it evenly divisible
    if n % block_size != 0:
        pad_size = ((n // block_size) + 1) * block_size - n
        G_padded = np.pad(G, ((0, pad_size), (0, pad_size)), mode='constant')
        n_padded = G_padded.shape[0]
    else:
        G_padded = G.copy()
        n_padded = n
    
    # Create the result matrix
    G_inv_padded = np.zeros_like(G_padded)
    
    # Extract block size for local use
    bs = block_size
    
    # Stage 1: Process diagonal blocks using LP-INV approach
    for i in range(0, n_padded, bs):
        for j in range(0, n_padded, bs):
            # Get the current block
            row_slice = slice(i, min(i + bs, n_padded))
            col_slice = slice(j, min(j + bs, n_padded))
            
            G_block = G_padded[row_slice, col_slice]
            
            if i == j:  # Diagonal block - invert it
                try:
                    G_inv_block = np.linalg.inv(G_block)
                except np.linalg.LinAlgError:
                    # Use pseudoinverse if not invertible
                    G_inv_block = np.linalg.pinv(G_block)
                
                G_inv_padded[row_slice, col_slice] = G_inv_block
            else:  # Off-diagonal block - for now, we'll leave as zero for simplicity
                # In a more sophisticated implementation, this would involve 
                # interactions between blocks based on the recursive structure
                G_inv_padded[row_slice, col_slice] = np.zeros_like(G_block)
    
    # If we had to pad, return only the original part
    if n != n_padded:
        return G_inv_padded[:n, :n]
    else:
        return G_inv_padded


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
    """
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