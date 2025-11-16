import asyncio
from typing import Optional, AsyncGenerator

from .forecast.forecast_result import ForecastResult
from utils.logging_config import get_logger

logger = get_logger(__name__)


class AlgoEngine:
    """AlgoEngine dedicated to producing forecast outputs for the new trading system."""

    def __init__(self, data_engine):
        self.data_engine = data_engine
        self.running = False
        self._last_forecast_timestamp: Optional[int] = None
        logger.info("AlgoEngine initialized for forecast strategies")

    async def run(self, strategy) -> AsyncGenerator[ForecastResult, None]:
        """Continuously execute the provided forecast strategy and yield forecasts."""
        if not hasattr(strategy, "calculate_forecast"):
            raise ValueError("Strategy must implement calculate_forecast() for forecast mode")

        if not self.running:
            self.running = True
            logger.info(f"Starting forecast strategy: {getattr(strategy, 'strategy_id', 'unknown')}")

        while self.running:
            try:
                forecast = await strategy.calculate_forecast()
                if forecast and forecast.timestamp != self._last_forecast_timestamp:
                    self._last_forecast_timestamp = forecast.timestamp
                    yield forecast
                await asyncio.sleep(1)
            except Exception as exc:
                logger.error(f"Error in forecast processing loop: {exc}")

    async def stop(self):
        self.running = False
        logger.info("AlgoEngine stopped")
