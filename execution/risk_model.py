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
        use_garch: bool = config.RISK_USE_GARCH,
        garch_params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.decay = decay
        self.shrinkage = shrinkage
        self.use_garch = use_garch
        self.garch_params = garch_params or config.GARCH_PARAMS
        self._ewma_state: Dict[str, float] = {}
        self._garch_state: Dict[str, float] = {}
        self._covariance: Optional[np.ndarray] = None
        self._symbols: List[str] = []
        self._last_mahalanobis: Optional[float] = None
        self._malv_history: List[float] = []
        self._cov_prev: Optional[np.ndarray] = None
        self._diag_realized_portfolio: Dict[str, float] = {}
        self._vol_loss_mse: Dict[str, List[float]] = {}
        self._vol_loss_qlike: Dict[str, List[float]] = {}

    def update(self, symbols: List[str], returns_matrix: np.ndarray) -> None:
        """Refresh covariance estimates given recent returns."""
        if returns_matrix.size == 0 or not symbols:
            logger.debug("RiskModel update skipped due to empty input")
            return

        if returns_matrix.shape[1] != len(symbols):
            raise ValueError("Returns matrix columns must match symbol list length")

        self._symbols = list(symbols)

        realized_cov_window = min(config.RISK_DIAG_WINDOW, returns_matrix.shape[0])
        realized_cov = None
        if realized_cov_window > 1:
            realized_cov = np.cov(returns_matrix[-realized_cov_window :].T, ddof=1)

        # Update EWMA variance per symbol using the most recent return
        latest_returns = returns_matrix[-1]
        for idx, symbol in enumerate(symbols):
            prev_var = self._ewma_state.get(symbol, np.var(returns_matrix[:, idx], ddof=1) or 1e-6)
            new_var = self.decay * prev_var + (1 - self.decay) * float(latest_returns[idx] ** 2)
            self._ewma_state[symbol] = max(new_var, 1e-10)
            if self.use_garch:
                self._update_garch(symbol, returns_matrix[:, idx])
            # Univariate vol loss diagnostics
            forecast_var = self._select_variance(symbol)
            realized_var = float(latest_returns[idx] ** 2)
            mse = (forecast_var - realized_var) ** 2
            qlike = np.log(max(forecast_var, 1e-12)) + realized_var / max(forecast_var, 1e-12)
            self._record_vol_loss(symbol, mse, qlike)

        # Sample covariance over the provided window
        window = min(config.RISK_COV_WINDOW, returns_matrix.shape[0])
        if window > 1:
            sample_cov = np.cov(returns_matrix[-window:].T, ddof=1)
            sample_cov = np.atleast_2d(sample_cov)
        else:
            sample_cov = np.diag([self._select_variance(s) for s in symbols])

        # Shrink towards diagonal target
        target = np.diag(np.diag(sample_cov))
        cov_new = self.shrinkage * sample_cov + (1 - self.shrinkage) * target
        self._cov_prev = self._covariance
        self._covariance = cov_new

        logger.info(
            "RiskModel refreshed | symbols=%d | window=%d",
            len(symbols),
            returns_matrix.shape[0],
        )

        # Mahalanobis diagnostic on latest return if covariance invertible
        try:
            inv_cov = np.linalg.pinv(self._covariance)
            d2 = float(latest_returns @ inv_cov @ latest_returns.T)
            self._last_mahalanobis = d2
            self._malv_history.append(d2 - len(symbols))
            if len(self._malv_history) > config.RISK_DIAG_WINDOW:
                self._malv_history = self._malv_history[-config.RISK_DIAG_WINDOW :]
        except Exception:
            self._last_mahalanobis = None

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

    def diagnostics(self) -> Dict[str, float]:
        """Return lightweight diagnostics for monitoring."""
        diag = {}
        if self._last_mahalanobis is not None:
            diag["last_mahalanobis_d2"] = self._last_mahalanobis
        if self._malv_history:
            diag["malv_mean"] = float(np.mean(self._malv_history))
        if self._cov_prev is not None and self._covariance is not None:
            diff = self._covariance - self._cov_prev
            diag["cov_turnover_fro"] = float(np.linalg.norm(diff, "fro"))
        # Aggregate vol losses
        if self._vol_loss_mse:
            diag["vol_mse_avg"] = float(
                np.mean([np.mean(v) for v in self._vol_loss_mse.values() if v])
            )
        if self._vol_loss_qlike:
            diag["vol_qlike_avg"] = float(
                np.mean([np.mean(v) for v in self._vol_loss_qlike.values() if v])
            )
        diag["symbols"] = len(self._symbols)
        return diag

    def covariance_loss_metrics(
        self, symbols: List[str], returns_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate covariance forecasts against realized cov using eigen and random portfolios."""
        metrics: Dict[str, float] = {}
        if self._covariance is None or returns_matrix.size == 0:
            return metrics
        window = min(config.RISK_DIAG_WINDOW, returns_matrix.shape[0])
        if window < 2:
            return metrics
        realized_cov = np.cov(returns_matrix[-window:].T, ddof=1)

        losses_mse = []
        losses_qlike = []

        def _eval(w: np.ndarray):
            w = w / (np.linalg.norm(w) + 1e-12)
            f_var = float(w @ self._covariance @ w)
            r_var = float(w @ realized_cov @ w)
            losses_mse.append((f_var - r_var) ** 2)
            if f_var > 0 and r_var > 0:
                losses_qlike.append(np.log(f_var) + r_var / f_var)

        try:
            eigvals, eigvecs = np.linalg.eigh(self._covariance)
            for w in eigvecs.T:
                _eval(w)
            # Random portfolios (Procedure 5.1 approximation)
            rng = np.random.default_rng(42)
            for _ in range(20):
                w = rng.normal(size=len(symbols))
                _eval(w)
        except Exception:
            pass

        if losses_mse:
            metrics["cov_port_mse"] = float(np.mean(losses_mse))
        if losses_qlike:
            metrics["cov_port_qlike"] = float(np.mean(losses_qlike))
        return metrics

    def _select_variance(self, symbol: str) -> float:
        """Choose variance estimate per symbol (GARCH if enabled and stable, else EWMA)."""
        ewma_var = self._ewma_state.get(symbol)
        if not self.use_garch:
            return ewma_var or 1e-10
        garch_var = self._garch_state.get(symbol)
        if garch_var is None:
            return ewma_var or 1e-10
        return max(min(garch_var, 1e2), 1e-10)

    def _update_garch(self, symbol: str, returns: np.ndarray) -> None:
        omega = self.garch_params.get("omega", 1e-6)
        alpha = self.garch_params.get("alpha", 0.05)
        beta = self.garch_params.get("beta", 0.9)
        if returns.size == 0:
            return
        # Initialize with variance of series
        var = float(np.var(returns, ddof=1) or 1e-6)
        for r in returns:
            var = omega + alpha * (r ** 2) + beta * var
        self._garch_state[symbol] = max(var, 1e-10)

    def _record_vol_loss(self, symbol: str, mse: float, qlike: float) -> None:
        self._vol_loss_mse.setdefault(symbol, []).append(mse)
        self._vol_loss_qlike.setdefault(symbol, []).append(qlike)
        if len(self._vol_loss_mse[symbol]) > config.RISK_DIAG_WINDOW:
            self._vol_loss_mse[symbol] = self._vol_loss_mse[symbol][-config.RISK_DIAG_WINDOW :]
        if len(self._vol_loss_qlike[symbol]) > config.RISK_DIAG_WINDOW:
            self._vol_loss_qlike[symbol] = self._vol_loss_qlike[symbol][-config.RISK_DIAG_WINDOW :]
