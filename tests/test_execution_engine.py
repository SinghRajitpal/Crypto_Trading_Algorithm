import asyncio
import numpy as np

import config
from algorithm.forecast.forecast_result import ForecastResult
from execution.execution_engine import ProductionExecutionEngine
from execution.trade_generator import OrderInstruction


class StubRiskModel:
    def __init__(self, cov):
        self.cov = cov

    def ready(self):
        return True

    def update(self, symbols, returns_matrix):
        return None

    def get_covariance(self, symbols):
        return self.cov

    def diagnostics(self):
        return {"diag": 1.0}

    def covariance_loss_metrics(self, symbols, returns_matrix):
        return {"cov_port_mse": 0.1}


class StubOptimizer:
    def __init__(self, weights):
        self.weights = weights

    def optimize(self, expected_returns, covariance, symbols, prev_weights=None):
        return dict(self.weights)


class StubTradeGenerator:
    def __init__(self, orders):
        self.orders = orders

    def generate_orders(self, **kwargs):
        return self.orders


class StubOrderExecutor:
    async def execute_orders(self, orders):
        return [
            {
                "status": "success",
                "symbol": o.symbol,
                "side": o.side,
                "quantity": o.quantity,
                # Simulate fills with slight adverse slippage to exercise metric calc
                "response": {"fills": [{"price": o.price * (1.001 if o.side == "buy" else 0.999), "qty": o.quantity}]},
                "slippage_bp": getattr(o, "slippage_bp", 0.0),
            }
            for o in orders
        ]


def _make_engine(cov, weights, orders):
    engine = ProductionExecutionEngine(binance_client=None, total_capital=1000.0)
    engine.risk_model = StubRiskModel(cov)
    engine.optimizer = StubOptimizer(weights)
    engine.trade_generator = StubTradeGenerator(orders)
    engine.order_executor = StubOrderExecutor()
    engine.current_weights = {}
    return engine


def test_process_forecast_skips_when_empty():
    engine = _make_engine(np.eye(1), {"A": 0.1}, [])
    forecast = ForecastResult(timestamp=0, universe=[], expected_returns={}, betas={}, diagnostics={})
    result = asyncio.run(engine.process_forecast(forecast, nav=1000.0, prices={}, returns_matrix=None))
    assert result["status"] == "skipped"
    assert result["reason"] == "empty forecast"


def test_process_forecast_skips_when_risk_not_ready(monkeypatch):
    engine = _make_engine(np.eye(1), {"A": 0.1}, [])

    class NotReady(StubRiskModel):
        def ready(self):
            return False

    engine.risk_model = NotReady(np.eye(1))
    forecast = ForecastResult(timestamp=0, universe=["A"], expected_returns={"A": 0.01}, betas={}, diagnostics={})
    result = asyncio.run(engine.process_forecast(forecast, nav=1000.0, prices={"A": 100.0}, returns_matrix=None))
    assert result["status"] == "skipped"
    assert "risk model not ready" in result["reason"]


def test_process_forecast_completes_and_updates_weights(monkeypatch):
    monkeypatch.setattr(config, "SLIPPAGE_BPS_DEFAULT", 5.0)
    symbols = ["A", "B"]
    cov = np.eye(2)
    target_weights = {"A": 0.2, "B": -0.1}
    orders = [
        OrderInstruction(symbol="A", side="buy", quantity=1.0, notional=100.0, price=100.0, target_weight=0.2, current_weight=0.0)
    ]
    engine = _make_engine(cov, target_weights, orders)
    forecast = ForecastResult(timestamp=0, universe=symbols, expected_returns={"A": 0.01, "B": 0.0}, betas={}, diagnostics={})
    prices = {"A": 100.0, "B": 50.0}
    returns_matrix = np.random.normal(scale=0.01, size=(config.RISK_WINDOW, 2))
    engine.refresh_risk_model(symbols, returns_matrix)

    result = asyncio.run(engine.process_forecast(forecast, nav=1000.0, prices=prices, returns_matrix=returns_matrix))
    assert result["status"] == "completed"
    expected_kelly = result["kelly_weights"]
    assert engine.current_weights == expected_kelly  # weights updated after success
    assert "risk_diag" in result and result["risk_diag"]["diag"] == 1.0
    assert "risk_cov_loss" in result and "cov_port_mse" in result["risk_cov_loss"]
    # Turnover computed vs Kelly-scaled weights (should be positive)
    assert result["turnover"] > 0
    assert result["expected_cost"] > 0
    assert result["expected_slippage"] > 0
    # Realized slippage reported
    assert result.get("realized_slippage_bp") is not None


def test_process_forecast_skips_when_no_rebalance():
    symbols = ["A"]
    cov = np.eye(1)
    target_weights = {"A": 0.0}
    engine = _make_engine(cov, target_weights, orders=[])
    forecast = ForecastResult(timestamp=0, universe=symbols, expected_returns={"A": 0.0}, betas={}, diagnostics={})
    prices = {"A": 100.0}
    returns_matrix = np.random.normal(scale=0.01, size=(config.RISK_WINDOW, 1))
    engine.refresh_risk_model(symbols, returns_matrix)
    result = asyncio.run(engine.process_forecast(forecast, nav=1000.0, prices=prices, returns_matrix=returns_matrix))
    assert result["status"] == "skipped"
    assert "portfolio already aligned" in result["reason"]


def test_slippage_overrides_increase_expected_cost(monkeypatch):
    symbols = ["A", "B"]
    cov = np.eye(2)
    target_weights = {"A": 0.2, "B": 0.0}
    engine = _make_engine(cov, target_weights, orders=[])
    base_cost = engine._expected_costs(target_weights, {}, nav=1000.0)
    monkeypatch.setattr(config, "SLIPPAGE_BPS_OVERRIDES", {"A": 20.0})
    higher_cost = engine._expected_costs(target_weights, {}, nav=1000.0)
    assert higher_cost["total_cost"] >= base_cost["total_cost"]
