from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np

import config
from algorithm.forecast.forecast_result import ForecastResult
from execution.risk_model import RiskModel
from execution.optimizer import MeanVarianceOptimizer
from execution.trade_generator import TradeGenerator
from execution.executor import OrderExecutor
from execution.kelly import compute_kelly_scaler, fractional_kelly_scaler, max_constraint_scaler
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
        self._nav_high_watermark: float = 0.0
        self._last_drawdown: float = 0.0
        self._pnl_history: list = []
        self._fee_rate = getattr(config, "FEE_RATE", 0.0)

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
        returns_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        prev_weights = dict(self.current_weights)
        if not forecast.expected_returns:
            return {"status": "skipped", "reason": "empty forecast"}

        if not self.risk_model.ready():
            return {"status": "skipped", "reason": "risk model not ready"}

        covariance = self.risk_model.get_covariance(forecast.universe)
        if covariance is None:
            return {"status": "skipped", "reason": "missing covariance for universe"}

        target_weights = self.optimizer.optimize(
            forecast.expected_returns,
            covariance,
            forecast.universe,
            prev_weights=prev_weights,
        )

        # Kelly overlay
        f_star, mu_p, var_p = compute_kelly_scaler(
            forecast.universe, target_weights, forecast.expected_returns, covariance
        )
        self._nav_high_watermark = max(self._nav_high_watermark, nav)
        drawdown = 0.0
        if self._nav_high_watermark > 0:
            drawdown = max(0.0, (self._nav_high_watermark - nav) / self._nav_high_watermark)
        self._last_drawdown = drawdown
        # Track pnl return for vol-aware Kelly dampening
        if self._pnl_history:
            self._pnl_history.append(nav)
        else:
            self._pnl_history.append(nav)
        realized_vol = self._realized_volatility()
        kelly_scaler = fractional_kelly_scaler(
            f_star,
            drawdown,
            lam_base=config.KELLY_FRACTION_BASE,
            thresholds=config.DRAWDOWN_THRESHOLDS,
            lambdas=config.DRAWDOWN_LAMBDAS,
            vol=realized_vol,
        )
        constraint_scaler = max_constraint_scaler(
            target_weights,
            config.WEIGHT_MIN,
            config.WEIGHT_MAX,
            config.MAX_GROSS_EXPOSURE,
            max_net=config.MAX_NET_EXPOSURE,
        )
        final_scaler = min(kelly_scaler, constraint_scaler)
        kelly_weights = {s: w * final_scaler for s, w in target_weights.items()}
        kelly_weights = self._net_and_bound_clamp(kelly_weights)
        kelly_weights = self._apply_cost_budget(kelly_weights, prev_weights, nav)
        portfolio_turnover = float(
            sum(abs(kelly_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in set(kelly_weights) | set(prev_weights))
        )

        orders = self.trade_generator.generate_orders(
            current_weights=prev_weights,
            target_weights=kelly_weights,
            nav=nav,
            prices=prices,
            precision_provider=self._get_symbol_filters,
        )

        if not orders:
            logger.debug(
                "No rebalancing needed | timestamp=%s | assets=%d",
                forecast.timestamp,
                len(forecast.universe),
            )
            return {
                "status": "skipped",
                "reason": "portfolio already aligned",
                "target_weights": target_weights,
            }

        total_notional = sum(order.notional for order in orders)
        logger.info(
            "Executing rebalance | orders=%d | total_notional=%.2f",
            len(orders),
            total_notional,
        )
        # Apply configured slippage per order (basis points)
        for order in orders:
            order.slippage_bp = self._slippage_bps(order.symbol)

        execution = await self.order_executor.execute_orders(orders)
        success = all(item["status"] == "success" for item in execution)

        if success:
            self.current_weights = kelly_weights
        else:
            failed = [item for item in execution if item["status"] != "success"]
            for item in failed:
                logger.error(
                    "Order failure | symbol=%s | side=%s | reason=%s",
                    item.get("symbol"),
                    item.get("side"),
                    item.get("reason"),
                )

        turnover_sigma = None
        if hasattr(self.risk_model, "_cov_prev") and self.risk_model._cov_prev is not None:
            alt_weights = self.optimizer.optimize(
                forecast.expected_returns,
                self.risk_model._cov_prev,
                forecast.universe,
                prev_weights=prev_weights,
            )
            turnover_sigma = float(
                sum(
                    abs(kelly_weights.get(s, 0.0) - alt_weights.get(s, 0.0))
                    for s in set(kelly_weights) | set(alt_weights)
                )
            )
        forecast_port_var = float(np.dot(np.array(list(kelly_weights.values())), covariance @ np.array(list(kelly_weights.values())))) if covariance is not None and kelly_weights else None
        realized_port_var = None
        if returns_matrix is not None and returns_matrix.size > 0 and kelly_weights:
            latest_r = returns_matrix[-1]
            w_vec = np.array([kelly_weights.get(sym, 0.0) for sym in forecast.universe], dtype=float)
            realized_port_var = float((latest_r @ w_vec) ** 2)
        cost_estimates = self._expected_costs(kelly_weights, prev_weights, nav)

        result = {
            "status": "completed" if success else "partial",
            "orders": execution,
            "target_weights": target_weights,
            "kelly_weights": kelly_weights,
            "kelly_f": f_star,
            "kelly_drawdown": drawdown,
            "kelly_mu": mu_p,
            "kelly_var": var_p,
            "kelly_scaler_raw": kelly_scaler,
            "kelly_scaler_constraints": constraint_scaler,
            "kelly_scaler_final": final_scaler,
            "turnover": portfolio_turnover,
            "turnover_sigma": turnover_sigma,
            "forecast_port_var": forecast_port_var,
            "realized_port_var": realized_port_var,
            "expected_cost": cost_estimates.get("total_cost"),
            "expected_cost_bp": cost_estimates.get("total_cost_bp"),
            "expected_fee": cost_estimates.get("fee_cost"),
            "expected_slippage": cost_estimates.get("slippage_cost"),
        }
        # Slippage diagnostics
        realized_slippage = self._compute_realized_slippage(execution, prices, nav)
        if realized_slippage is not None:
            result["realized_slippage_bp"] = realized_slippage
        if returns_matrix is not None:
            result["risk_diag"] = self.risk_model.diagnostics()
            result["risk_cov_loss"] = self.risk_model.covariance_loss_metrics(
                forecast.universe, returns_matrix
            )
        return result

    def _get_symbol_filters(self, symbol: str) -> Optional[Dict[str, float]]:
        try:
            return self.binance_client.get_symbol_filters(symbol)
        except Exception as exc:
            logger.warning("Symbol filter lookup failed for %s: %s", symbol, exc)
            return None

    @staticmethod
    def _compute_realized_slippage(execution: Any, prices: Dict[str, float], nav: float) -> Optional[float]:
        """Compute realized slippage in basis points vs mid (approx using provided prices)."""
        if not execution:
            return None
        total_notional = 0.0
        total_slip = 0.0
        for order in execution:
            sym = order.get("symbol")
            qty = order.get("quantity", 0.0)
            side = order.get("side")
            resp = order.get("response") or {}
            fills = resp.get("fills") if isinstance(resp, dict) else None
            if fills and sym and prices.get(sym):
                avg_fill = sum(float(f.get("price", 0.0)) * float(f.get("qty", 0.0)) for f in fills) / max(
                    1e-9, sum(float(f.get("qty", 0.0)) for f in fills)
                )
                mid = prices[sym]
                slip = (avg_fill - mid) / mid * (1 if side == "buy" else -1)
                notional = qty * mid
                total_notional += abs(notional)
                total_slip += slip * abs(notional)
        if total_notional == 0:
            return None
        # Basis points weighted by notional
        return float((total_slip / total_notional) * 10000)

    def _slippage_bps(self, symbol: str) -> float:
        return float(config.SLIPPAGE_BPS_OVERRIDES.get(symbol, config.SLIPPAGE_BPS_DEFAULT))

    def _expected_costs(self, target_weights: Dict[str, float], prev_weights: Dict[str, float], nav: float) -> Dict[str, float]:
        """Estimate expected fee + slippage cost for moving from prev to target weights."""
        fee_rate = max(self._fee_rate, 0.0)
        total_fee = 0.0
        total_slip = 0.0
        symbols = set(target_weights) | set(prev_weights)
        for sym in symbols:
            delta_w = target_weights.get(sym, 0.0) - prev_weights.get(sym, 0.0)
            notional = abs(delta_w * nav)
            if notional <= 0:
                continue
            bps = self._slippage_bps(sym)
            total_slip += notional * (bps / 10000.0)
            total_fee += notional * fee_rate
        total_cost = total_fee + total_slip
        cost_bp = (total_cost / nav * 10000.0) if nav > 0 else None
        return {
            "total_cost": float(total_cost),
            "total_cost_bp": float(cost_bp) if cost_bp is not None else None,
            "fee_cost": float(total_fee),
            "slippage_cost": float(total_slip),
        }

    def _apply_cost_budget(self, target_weights: Dict[str, float], prev_weights: Dict[str, float], nav: float) -> Dict[str, float]:
        """Scale weight changes if expected cost exceeds configured budget (in bp of NAV)."""
        budget_bp = config.COST_BUDGET_BP
        if budget_bp is None or budget_bp <= 0 or nav <= 0:
            return target_weights
        costs = self._expected_costs(target_weights, prev_weights, nav)
        cost_bp = costs.get("total_cost_bp")
        if cost_bp is None or cost_bp <= budget_bp:
            return target_weights
        scale = max(0.0, min(1.0, budget_bp / cost_bp))
        scaled = {}
        symbols = set(target_weights) | set(prev_weights)
        for sym in symbols:
            prev = prev_weights.get(sym, 0.0)
            tgt = target_weights.get(sym, 0.0)
            scaled[sym] = prev + (tgt - prev) * scale
        return self._net_and_bound_clamp(scaled)

    def _net_and_bound_clamp(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply lightweight net-exposure clamp and per-asset bounds."""
        if not weights:
            return {}
        vals = list(weights.values())
        net = sum(vals)
        adj_weights = dict(weights)
        if abs(net) > config.MAX_NET_EXPOSURE and len(weights) > 0:
            adjustment = net - np.sign(net) * config.MAX_NET_EXPOSURE
            delta = adjustment / len(weights)
            for k in adj_weights:
                adj_weights[k] -= delta
        for k, v in adj_weights.items():
            adj_weights[k] = float(np.clip(v, config.WEIGHT_MIN, config.WEIGHT_MAX))
        return adj_weights

    def _realized_volatility(self) -> float:
        """Compute simple realized volatility of NAV changes."""
        if len(self._pnl_history) < 5:
            return 0.0
        nav_series = np.array(self._pnl_history[-config.RISK_DIAG_WINDOW :], dtype=float)
        returns = np.diff(nav_series) / nav_series[:-1]
        if returns.size == 0:
            return 0.0
        return float(np.std(returns, ddof=1))


# Backward-compatible alias for backtesting imports
ExecutionEngine = ProductionExecutionEngine
