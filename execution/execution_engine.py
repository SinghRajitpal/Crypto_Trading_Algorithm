from __future__ import annotations

from typing import Dict, Any, List

import numpy as np

from algorithm.forecast.forecast_result import ForecastResult
from execution.risk_model import RiskModel
from execution.optimizer import MeanVarianceOptimizer
from execution.trade_generator import TradeGenerator
from execution.executor import OrderExecutor
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ProductionExecutionEngine:
    """Execution engine consuming forecasts to produce and execute portfolio trades."""

    def __init__(self, binance_client, total_capital: float = 0.0) -> None:
        self.binance_client = binance_client
        self.total_capital = total_capital
        self.risk_model = RiskModel()
        self.optimizer = MeanVarianceOptimizer()
        self.trade_generator = TradeGenerator()
        self.order_executor = OrderExecutor(binance_client)
        self.current_weights: Dict[str, float] = {}

    def update_total_capital(self, total_capital: float) -> None:
        self.total_capital = total_capital

    def refresh_risk_model(self, symbols: List[str], returns_matrix: np.ndarray) -> None:
        self.risk_model.update(symbols, returns_matrix)

    def get_current_weights(self) -> Dict[str, float]:
        return dict(self.current_weights)

    async def process_forecast(
        self,
        forecast: ForecastResult,
        nav: float,
        prices: Dict[str, float],
    ) -> Dict[str, Any]:
        if not forecast.expected_returns:
            return {"status": "skipped", "reason": "empty forecast"}

        if not self.risk_model.ready():
            return {"status": "skipped", "reason": "risk model not ready"}

        covariance = self.risk_model.get_covariance(forecast.universe)
        if covariance is None:
            return {"status": "skipped", "reason": "missing covariance for universe"}

        target_weights = self.optimizer.optimize(
            forecast.expected_returns, covariance, forecast.universe
        )

        orders = self.trade_generator.generate_orders(
            current_weights=self.current_weights,
            target_weights=target_weights,
            nav=nav,
            prices=prices,
        )

        if not orders:
            return {"status": "skipped", "reason": "portfolio already aligned", "target_weights": target_weights}

        execution = await self.order_executor.execute_orders(orders)
        success = all(item["status"] == "success" for item in execution)

        if success:
            self.current_weights = target_weights

        return {
            "status": "completed" if success else "partial",
            "orders": execution,
            "target_weights": target_weights,
        }
