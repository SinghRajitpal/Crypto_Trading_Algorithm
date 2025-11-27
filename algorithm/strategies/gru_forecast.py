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
            window = self._build_window(sym)
            if window is None:
                continue
            try:
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
        """Construct the latest window of log-return/log-volume features."""
        lr_hist = self.data_engine.return_manager.log_return_history.get(symbol, [])
        vol_hist = self.data_engine.return_manager.volume_history.get(symbol, [])
        if len(lr_hist) < self.lookback or len(vol_hist) < self.lookback:
            return None
        log_returns = np.array([v for _, v in lr_hist], dtype=float)
        volumes = np.array([v for _, v in vol_hist], dtype=float)
        # Align volume length to log return length if needed
        volumes = volumes[-len(log_returns) :]
        lr_slice = log_returns[-self.lookback :]
        vol_slice = volumes[-self.lookback :]
        if lr_slice.shape[0] < self.lookback or vol_slice.shape[0] < self.lookback:
            return None
        log_vol = np.log1p(vol_slice)
        window = np.stack([lr_slice, log_vol], axis=1)
        if not np.all(np.isfinite(window)):
            return None
        return window
