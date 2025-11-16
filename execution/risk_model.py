from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RiskModel:
    """Maintains per-asset volatilities and covariance matrix for the optimizer."""

    def __init__(
        self,
        decay: float = config.EWMA_LAMBDA,
        shrinkage: float = config.COVARIANCE_SHRINKAGE,
    ) -> None:
        self.decay = decay
        self.shrinkage = shrinkage
        self._ewma_state: Dict[str, float] = {}
        self._covariance: Optional[np.ndarray] = None
        self._symbols: List[str] = []

    def update(self, symbols: List[str], returns_matrix: np.ndarray) -> None:
        """Refresh covariance estimates given recent returns."""
        if returns_matrix.size == 0 or not symbols:
            logger.debug("RiskModel update skipped due to empty input")
            return

        if returns_matrix.shape[1] != len(symbols):
            raise ValueError("Returns matrix columns must match symbol list length")

        self._symbols = list(symbols)

        # Update EWMA variance per symbol using the most recent return
        latest_returns = returns_matrix[-1]
        for idx, symbol in enumerate(symbols):
            prev_var = self._ewma_state.get(symbol, np.var(returns_matrix[:, idx], ddof=1) or 1e-6)
            new_var = self.decay * prev_var + (1 - self.decay) * float(latest_returns[idx] ** 2)
            self._ewma_state[symbol] = max(new_var, 1e-10)

        # Sample covariance over the provided window
        if returns_matrix.shape[0] > 1:
            sample_cov = np.cov(returns_matrix.T, ddof=1)
        else:
            sample_cov = np.diag([self._ewma_state[s] for s in symbols])

        # Shrink towards diagonal target
        target = np.diag(np.diag(sample_cov))
        self._covariance = (
            self.shrinkage * sample_cov + (1 - self.shrinkage) * target
        )

        logger.info(
            "RiskModel refreshed | symbols=%d | window=%d",
            len(symbols),
            returns_matrix.shape[0],
        )

    def get_covariance(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Return the covariance matrix aligned to the requested symbol order."""
        if self._covariance is None or not self._symbols:
            return None

        index_map = {symbol: idx for idx, symbol in enumerate(self._symbols)}
        try:
            rows = [index_map[symbol] for symbol in symbols]
        except KeyError as exc:
            logger.warning("RiskModel missing symbol for covariance request: %s", exc)
            return None

        cov = self._covariance[np.ix_(rows, rows)]
        return cov

    def get_volatility(self, symbol: str) -> Optional[float]:
        """Return the EWMA volatility estimate for a symbol."""
        variance = self._ewma_state.get(symbol)
        if variance is None:
            return None
        return float(np.sqrt(max(variance, 1e-10)))

    def ready(self) -> bool:
        return self._covariance is not None
