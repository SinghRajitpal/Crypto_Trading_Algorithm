import time
from typing import Dict, Optional

import config
from utils.logging_config import get_logger

from algorithm.forecast.forecast_result import ForecastResult
from algorithm.forecast.linear_regression import LinearRegressionForecaster

logger = get_logger(__name__)


class MeanVarianceForecastStrategy:
    """Strategy that produces expected returns for mean–variance optimization."""

    strategy_id = "mean_variance_regression"

    def __init__(
        self,
        data_engine,
        forecaster: Optional[LinearRegressionForecaster] = None,
        min_history: Optional[int] = None,
    ):
        self.data_engine = data_engine
        self.forecaster = forecaster or LinearRegressionForecaster(
            window=config.REGRESSION_WINDOW
        )
        self.min_history = min_history or config.REGRESSION_WINDOW
        self.timeframe = getattr(data_engine, "primary_timeframe", config.PRIMARY_TIMEFRAME)

    async def calculate_forecast(self) -> Optional[ForecastResult]:
        """Collect the latest data, run regressions, and return expected returns."""
        self.data_engine.process_all_latest_bars(self.timeframe)
        universe = self.data_engine.get_active_universe()
        if not universe:
            logger.debug("[MeanVarianceForecast] No active universe available")
            return None

        expected_returns: Dict[str, float] = {}
        betas: Dict[str, Dict[str, float]] = {}
        diagnostics: Dict[str, Dict[str, float]] = {}
        latest_timestamp = 0

        for symbol in universe:
            features = self.data_engine.get_feature_series(symbol, self.forecaster.window + 5)
            returns = self.data_engine.get_return_series(symbol, self.forecaster.window + 5)

            if len(returns) < self.min_history or len(features) < self.min_history:
                logger.debug(f"[MeanVarianceForecast] Insufficient history for {symbol}")
                continue

            regression = self.forecaster.forecast(symbol, features, returns)
            if not regression:
                continue

            expected_returns[symbol] = regression.expected_return
            betas[symbol] = {
                "beta0": regression.beta0,
                "beta1": regression.beta1,
                "samples": regression.sample_count,
            }
            diagnostics[symbol] = {"r_squared": regression.r_squared}

            candle = self.data_engine.get_latest_candle(symbol, self.timeframe)
            if candle:
                latest_timestamp = max(latest_timestamp, int(candle[0]))

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
            },
        )
