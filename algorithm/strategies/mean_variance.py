import time
from typing import Dict, Optional
from collections import defaultdict, deque

import config
from utils.logging_config import get_logger

from algorithm.forecast.forecast_result import ForecastResult
from algorithm.forecast.ridge_regression import RidgeRegressionForecaster

logger = get_logger(__name__)


class MeanVarianceForecastStrategy:
    """Strategy that produces expected returns for mean–variance optimization."""

    strategy_id = "mean_variance_regression"

    def __init__(
        self,
        data_engine,
        forecaster: Optional[RidgeRegressionForecaster] = None,
        min_history: Optional[int] = None,
    ):
        self.data_engine = data_engine
        self.forecaster = forecaster or RidgeRegressionForecaster()
        self.min_history = min_history or config.REGRESSION_WINDOW
        self.timeframe = getattr(data_engine, "primary_timeframe", config.PRIMARY_TIMEFRAME)
        self._pending_forecasts: Dict[str, Dict[str, float]] = {}
        self._error_history = defaultdict(lambda: deque(maxlen=500))
        self._ic_history = defaultdict(lambda: deque(maxlen=500))

    async def calculate_forecast(self) -> Optional[ForecastResult]:
        """Collect the latest data, run regressions, and return expected returns."""
        self.data_engine.process_all_latest_bars(self.timeframe)
        self._update_forecast_metrics()
        universe = self.data_engine.get_active_universe()
        if not universe:
            logger.debug("[MeanVarianceForecast] No active universe available")
            return None

        expected_returns: Dict[str, float] = {}
        betas: Dict[str, Dict[str, float]] = {}
        diagnostics: Dict[str, Dict[str, float]] = {}
        latest_timestamp = 0

        for symbol in universe:
            # Skip symbols with missing bars detected
            if self.data_engine.get_missing_bars(symbol, self.timeframe):
                continue
            X, y, ts, cols = self.data_engine.get_feature_matrix(symbol)
            if y.size < self.min_history:
                logger.debug(f"[MeanVarianceForecast] Insufficient history for {symbol}")
                continue

            ridge_result = self.forecaster.forecast(symbol, X, y)
            if not ridge_result:
                continue

            expected_returns[symbol] = ridge_result.expected_simple_return
            betas[symbol] = {
                "k_best": ridge_result.k_best,
                "samples": ridge_result.samples,
                "dropped_features": len(ridge_result.dropped_features),
            }
            diagnostics[symbol] = {
                "msep": ridge_result.msep,
                "gcv": ridge_result.gcv,
                "rl_vs_ls": ridge_result.rl_vs_ls if ridge_result.rl_vs_ls is not None else float("nan"),
                "t_threshold": ridge_result.t_threshold,
                "rolling_msep": self._current_msep(symbol),
                "rolling_ic": self._current_ic(symbol),
                "hat_mean": ridge_result.hat_mean,
                "hat_max": ridge_result.hat_max,
                "resid_sigma": ridge_result.resid_sigma,
            }

            candle = self.data_engine.get_latest_candle(symbol, self.timeframe)
            if candle:
                latest_timestamp = max(latest_timestamp, int(candle[0]))
                target_ts = int(candle[0]) + self._timeframe_ms(self.timeframe)
                self._pending_forecasts[symbol] = {
                    "target_ts": target_ts,
                    "pred_log_ret": ridge_result.expected_log_return,
                }

        if not expected_returns:
            logger.debug("[MeanVarianceForecast] No forecasts produced this cycle")
            return None

        timestamp = latest_timestamp or int(time.time() * 1000)
        return ForecastResult(
            timestamp=timestamp,
            universe=list(expected_returns.keys()),
            expected_returns=expected_returns,
            betas=betas,
            diagnostics={
                "model": "linear_regression",
                "per_symbol": diagnostics,
                "forecast_monitor": {
                    sym: {
                        "rolling_msep": self._current_msep(sym),
                        "rolling_ic": self._current_ic(sym),
                        "samples": len(self._error_history[sym]),
                    }
                    for sym in expected_returns.keys()
                },
            },
        )

    def _update_forecast_metrics(self) -> None:
        """Update rolling forecast errors using realized log returns."""
        to_remove = []
        for sym, info in list(self._pending_forecasts.items()):
            latest = self.data_engine.return_manager.get_latest_log_return_with_ts(sym)
            if not latest:
                continue
            ts, realized = latest
            target_ts = info.get("target_ts")
            if target_ts and ts >= target_ts:
                err = realized - info.get("pred_log_ret", 0.0)
                self._error_history[sym].append(err**2)
                self._ic_history[sym].append((realized, info.get("pred_log_ret", 0.0)))
                to_remove.append(sym)
        for sym in to_remove:
            self._pending_forecasts.pop(sym, None)

    def _current_msep(self, symbol: str) -> float:
        hist = self._error_history.get(symbol)
        if not hist:
            return float("nan")
        return float(sum(hist) / len(hist))

    def _current_ic(self, symbol: str) -> float:
        hist = self._ic_history.get(symbol)
        if not hist:
            return float("nan")
        realized = np.array([r for r, p in hist])
        preds = np.array([p for r, p in hist])
        if realized.size < 2 or preds.size < 2:
            return float("nan")
        cov = np.cov(realized, preds, ddof=1)
        if cov.shape != (2, 2) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
            return float("nan")
        return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

    @staticmethod
    def _timeframe_ms(tf: str) -> int:
        if not tf.endswith("m"):
            return 0
        return int(tf[:-1]) * 60 * 1000
