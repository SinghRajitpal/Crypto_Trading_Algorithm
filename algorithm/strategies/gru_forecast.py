"""GRU-based forecast strategy for producing expected returns."""

from __future__ import annotations

import time
import numpy as np

import config
from algorithm.forecast.forecast_result import ForecastResult
from algorithm.forecast.gru_torch import GRUForecasterTorch
from utils.logging_config import get_logger

logger = get_logger(__name__)


class GRUForecastStrategy:
    """Strategy that uses a pre-trained GRU to forecast next-bar returns."""

    strategy_id = "gru_forecast"

    def __init__(self, data_engine, model_dir: str, symbols=None, timeframe: str = None):
        self.data_engine = data_engine
        self.model_dir = model_dir
        self.forecaster = GRUForecasterTorch.load(model_dir, device=config.GRU_DEVICE)
        self.lookback = self.forecaster.lookback
        self.timeframe = timeframe or config.GRU_TIMEFRAME
        self.symbols = symbols

    async def calculate_forecast(self) -> ForecastResult | None:
        self.data_engine.process_all_latest_bars(self.timeframe)
        universe = self.symbols or self.data_engine.get_active_universe()
        expected_returns = {}
        for sym in universe:
            if self.data_engine.get_missing_bars(sym, self.timeframe):
                continue
            if hasattr(self.data_engine, "feature_ready") and not self.data_engine.feature_ready(sym, self.lookback):
                logger.debug("Features not ready for %s | lookback=%d", sym, self.lookback)
                continue
            window = self._build_window(sym)
            if window is None:
                logger.debug("No feature window for %s | lookback=%d", sym, self.lookback)
                continue
            try:
                if not np.all(np.isfinite(window)):
                    logger.warning("Non-finite values in feature window for %s; skipping", sym)
                    continue
                if getattr(self.forecaster, "feature_schema", None):
                    expected = len(self.forecaster.feature_schema)
                    if window.shape[1] != expected:
                        logger.warning(
                            "Feature window width mismatch for %s | expected=%d got=%d",
                            sym,
                            expected,
                            window.shape[1],
                        )
                        continue
                mu = self.forecaster.predict_simple_return(window)
                expected_returns[sym] = mu
            except Exception as exc:
                logger.warning("GRU forecast failed for %s: %s", sym, exc)
                continue
        if not expected_returns:
            return None
        latest_ts = 0
        for sym in expected_returns.keys():
            candle = self.data_engine.get_latest_candle(sym, self.timeframe)
            if candle:
                latest_ts = max(latest_ts, int(candle[0]))
        timestamp = latest_ts or int(time.time() * 1000)
        return ForecastResult(
            timestamp=timestamp,
            universe=list(expected_returns.keys()),
            expected_returns=expected_returns,
            betas={},
            diagnostics={"model": "gru"},
        )

    def _build_window(self, symbol: str):
        """Construct the latest window of pre-computed features."""
        window = self.data_engine.get_feature_window(symbol, self.lookback)
        if window is None:
            return None
        # Validate schema alignment if available
        if getattr(self.forecaster, "feature_schema", None):
            expected = len(self.forecaster.feature_schema)
            if window.shape[1] != expected:
                logger.warning(
                    "Feature window width mismatch for %s | expected=%d got=%d",
                    symbol,
                    expected,
                    window.shape[1],
                )
                return None
        return window
