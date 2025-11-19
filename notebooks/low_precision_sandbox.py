import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

def quantize(matrix: np.ndarray, bits: int = 3) -> np.ndarray:
    """Quantize matrix to given bit precision."""
    if matrix.size == 0 or np.all(matrix == 0):
        return matrix
    max_val = np.max(np.abs(matrix))
    levels = 2**bits - 1
    scale = levels / max_val
    quantized = np.round(matrix * scale) / scale
    return quantized

def lp_inv(A: np.ndarray, bits: int = 3, noise_std: float = 0.01) -> np.ndarray:
    """Compute low-precision inverse with quantization and noise."""
    A_q = quantize(A, bits)
    try:
        A_inv_q = np.linalg.inv(A_q)
        noise = np.random.normal(0, noise_std, A_inv_q.shape)
        return A_inv_q + noise
    except np.linalg.LinAlgError:
        return np.zeros_like(A)

def generate_well_conditioned_matrix(n: int, kappa: float = 5.0) -> np.ndarray:
    """Generate a random matrix with condition number approximately kappa."""
    # Create orthogonal matrix
    U = np.random.randn(n, n)
    U, _ = np.linalg.qr(U)
    # Create diagonal with eigenvalues from 1 to kappa
    D = np.linspace(1, kappa, n)
    A = U @ np.diag(D) @ U.T
    return A

def simulate_drift(n: int = 8, kappa: float = 5.0, bits: int = 3, noise_std: float = 0.01, max_iter: int = 10):
    """Simulate the drift of low-precision analogue inversion."""
    np.random.seed(42)  # For reproducibility
    A = generate_well_conditioned_matrix(n, kappa)
    b = np.random.randn(n)

    # Compute true solution
    x_true = np.linalg.solve(A, b)

    # Low-precision inverse
    A_lp_inv = lp_inv(A, bits, noise_std)

    # Iterative refinement
    x = np.zeros_like(b)
    residuals = []
    for k in range(max_iter):
        r = b - A @ x
        residuals.append(np.linalg.norm(r))
        delta = A_lp_inv @ r
        x += delta

    # Plot log2 of residual norm
    plt.figure(figsize=(8, 6))
    plt.plot(range(max_iter), np.log2(residuals), marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('log₂(||r(k)||)')
    plt.title(f'Low-Precision Inversion Drift (n={n}, κ≈{kappa}, bits={bits})')
    plt.grid(True)
    plt.savefig('notebooks/low_precision_drift.png', dpi=150)
    # plt.show()  # Commented for headless

    print(f"Initial residual: {residuals[0]:.2e}")
    print(f"Final residual: {residuals[-1]:.2e}")
    print(f"True solution norm: {np.linalg.norm(x_true):.2e}")
    print(f"Approximate solution error: {np.linalg.norm(x - x_true):.2e}")
    print("Plot saved as low_precision_drift.png")

if __name__ == "__main__":
    simulate_drift()