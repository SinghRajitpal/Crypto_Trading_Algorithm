import numpy as np
from typing import Tuple


def ras_sharpe(returns_matrix: np.ndarray, delta: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RAS haircuts for Sharpe estimates across strategies.

    Simplified bound: H = sqrt(2 log(2/delta) / T)
    Returns (empirical_sharpe, lower_bounds)
    """
    if returns_matrix.size == 0:
        return np.array([]), np.array([])
    T = returns_matrix.shape[1]
    if T == 0:
        return np.array([]), np.array([])
    sharpe = np.mean(returns_matrix, axis=1)
    haircut = np.sqrt(2 * np.log(2 / delta) / T)
    lower = sharpe - haircut
    return sharpe, lower


def ras_ic(ic_matrix: np.ndarray, delta: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """RAS for information coefficients (treated like returns)."""
    return ras_sharpe(ic_matrix, delta=delta)
