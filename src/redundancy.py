import numpy as np

def apply_redundancy(G: np.ndarray) -> np.ndarray:
    """
    Apply basic redundancy scheme: detect stuck-at faults (e.g., zero conductance)
    and replace faulty rows/columns with averages of adjacent ones.

    Args:
        G: Conductance matrix

    Returns:
        Repaired matrix
    """
    G_repaired = G.copy()
    n = G.shape[0]

    # Detect faulty cells (e.g., exactly zero, assuming stuck-at-0)
    faulty = (G == 0)

    for i in range(n):
        for j in range(n):
            if faulty[i, j]:
                # Replace with average of non-faulty neighbors
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and not faulty[ni, nj]:
                        neighbors.append(G[ni, nj])
                if neighbors:
                    G_repaired[i, j] = np.mean(neighbors)
                # Else leave as is or set to small value

    return G_repaired