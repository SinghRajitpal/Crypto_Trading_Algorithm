from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

import config
from algorithm.forecast.gru_torch import GRUForecasterTorch
from algorithm.forecast.forecast_result import ForecastResult
from backtest.broker import SimBroker
from backtest.ras import ras_sharpe
from backtest.ras import ras_ic
from data.data_engine import DataEngine
from data.historical_data import HistoricalDataFetcher
from execution.execution_engine import ExecutionEngine
from utils.logging_config import console_log, get_logger
from backtest.visualization import (
    plot_equity,
    plot_risk_diagnostics,
    plot_series,
    plot_scatter,
    save_summary_json,
)

logger = get_logger(__name__)


@dataclass
class BacktestMetrics:
    pnl: float
    cum_return_pct: float
    sharpe: float
    max_drawdown: float
    turnover: float
    winrate: float
    impact_vs_slippage: float
    ras_sharpe: float = float("nan")
    ras_sharpe_lower: float = float("nan")


class WalkForwardBacktester:
    """Layer B walk-forward backtest using live components with per-asset retraining."""

    def __init__(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        gru_model_dir: str,
        initial_capital: float = 10_000.0,
        output_dir: Optional[str] = None,
    ) -> None:
        if not gru_model_dir:
            raise ValueError("gru_model_dir is required for GRU backtesting.")
        self.symbols = symbols
        self.start = start
        self.end = end
        self.gru_model_dir = gru_model_dir
        self.gru_forecaster: Optional[GRUForecasterTorch] = None
        self.initial_capital = initial_capital
        self.output_dir = output_dir
        self.timeframe = config.GRU_TIMEFRAME

        self.broker = SimBroker(initial_capital)
        self.data_engine = DataEngine(binance_client=self.broker, max_candles=2000)
        self.data_engine.primary_timeframe = self.timeframe
        self.data_engine.data_fetcher.symbol_timeframes = [(s, self.timeframe) for s in symbols]
        self.execution_engine = ExecutionEngine(binance_client=self.broker, total_capital=initial_capital)
        self.fetcher = HistoricalDataFetcher(testnet=False)
        self._metrics: Dict[str, Any] = {}
        # Use latest processed price for simulated fills
        self.broker.set_price_callback(self._price_lookup)

    async def run(self) -> BacktestMetrics:
        ohlcv_data = await self._load_history()
        await self._warmup(ohlcv_data)
        if self.gru_model_dir:
            self.gru_forecaster = GRUForecasterTorch.load(self.gru_model_dir, device=config.GRU_DEVICE)
        clock = sorted({ts for df in ohlcv_data.values() for ts in df.index})

        equity_curve: List[float] = []
        returns: List[float] = []
        turnovers: List[float] = []
        impact_vs_slippage: List[float] = []
        wins = 0
        trades = 0
        risk_diags: List[Dict[str, Any]] = []
        cov_losses: List[Dict[str, Any]] = []
        drawdowns: List[float] = []
        peak = self.initial_capital
        benchmark_prices: List[float] = []
        kelly_fs: List[float] = []
        kelly_drawdowns: List[float] = []
        impact_estimates: List[float] = []
        slippages: List[float] = []
        risk_series: List[Dict[str, Any]] = []

        for ts in clock:
            if ts < self.start or ts > self.end:
                continue

            # Align simulated broker timestamps with bar time
            self.broker.set_bar_timestamp(ts)

            for sym in self.symbols:
                df = ohlcv_data.get(sym)
                if df is not None and ts in df.index:
                    candle = df.loc[ts]
                    bar = [int(ts.timestamp() * 1000)] + candle.tolist()
                    await self.data_engine.data_fetcher.data_processor.update_tracked_candles(
                        sym, self.timeframe, bar
                    )

            self.data_engine.process_all_latest_bars(self.timeframe)

            # Mark-to-market: cache latest closes for all tracked symbols so equity reflects price moves even without trades.
            tracked_syms = set(self.symbols) | set(self.broker._positions.keys())  # pylint: disable=protected-access
            price_map = {
                s: self.data_engine.get_latest_price(s, self.timeframe)
                for s in tracked_syms
            }
            self.broker.update_last_prices(price_map)
            logger.debug(
                "Bar %s | tracked_syms=%d | positions=%d",
                ts.isoformat(),
                len(tracked_syms),
                len(self.broker._positions),  # pylint: disable=protected-access
            )

            expected_returns = self._predict_with_gru()
            logger.debug(
                "Bar %s | forecasts=%d | universe=%s",
                ts.isoformat(),
                len(expected_returns),
                list(expected_returns.keys()),
            )

            if not expected_returns:
                continue

            diagnostics = {"model": "gru"} if self.gru_forecaster else {}
            forecast = ForecastResult(
                timestamp=int(ts.timestamp() * 1000),
                universe=list(expected_returns.keys()),
                expected_returns=expected_returns,
                betas={},
                diagnostics=diagnostics,
            )

            returns_matrix, sym_order = self.data_engine.get_return_matrix(
                forecast.universe, config.RISK_WINDOW
            )
            if returns_matrix.size == 0 or not sym_order:
                continue

            self.execution_engine.refresh_risk_model(sym_order, returns_matrix)
            prices = {s: self.data_engine.get_latest_price(s, self.timeframe) for s in forecast.universe}
            nav = await self._nav()
            result = await self.execution_engine.process_forecast(
                forecast=forecast,
                nav=nav,
                prices=prices,
                returns_matrix=returns_matrix,
            )
            logger.debug(
                "Bar %s | nav=%.2f | orders=%d | turnover=%.4f",
                ts.isoformat(),
                nav,
                len(result.get("orders", [])),
                result.get("turnover", 0.0),
            )

            nav_new = await self._nav()
            equity_curve.append(nav_new)
            peak = max(peak, nav_new)
            dd = (peak - nav_new) / peak if peak > 0 else 0.0
            drawdowns.append(dd)
            # Benchmark: use first symbol close
            bench_sym = self.symbols[0] if self.symbols else None
            if bench_sym:
                px = self.data_engine.get_latest_price(bench_sym, self.timeframe)
                if px:
                    benchmark_prices.append(px)
            if len(equity_curve) > 1:
                r = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                returns.append(r)
                trades += len(result.get("orders", []))
                if r > 0:
                    wins += 1
            turnovers.append(result.get("turnover", 0.0))
            if "impact_vs_slippage" in result and result.get("impact_vs_slippage") is not None:
                impact_vs_slippage.append(result.get("impact_vs_slippage"))
            if result.get("risk_diag"):
                risk_diags.append(result.get("risk_diag"))
            if result.get("risk_cov_loss"):
                cov_losses.append(result.get("risk_cov_loss"))
            # Capture time-series risk/impact
            risk_series.append(
                {
                    "ts": forecast.timestamp,
                    "risk_diag": result.get("risk_diag"),
                    "risk_cov_loss": result.get("risk_cov_loss"),
                    "turnover": result.get("turnover"),
                    "impact_est": result.get("impact_cost_est"),
                    "impact_concave": result.get("impact_cost_concave"),
                    "impact_propagator": result.get("impact_cost_propagator"),
                }
            )
            if result.get("kelly_f") is not None:
                kelly_fs.append(result.get("kelly_f"))
            if result.get("kelly_drawdown") is not None:
                kelly_drawdowns.append(result.get("kelly_drawdown"))
            if result.get("impact_cost_est") is not None:
                impact_estimates.append(result.get("impact_cost_est"))
            if result.get("realized_slippage_bp") is not None:
                slippages.append(result.get("realized_slippage_bp"))

        pnl = equity_curve[-1] - self.initial_capital if equity_curve else 0.0
        cum_return_pct = (equity_curve[-1] / self.initial_capital - 1) * 100 if equity_curve else 0.0
        sharpe = self._sharpe(returns)
        max_dd = self._max_drawdown(equity_curve)
        turnover = float(np.mean(turnovers)) if turnovers else 0.0
        winrate = wins / max(1, trades)
        ivs = float(np.mean(impact_vs_slippage)) if impact_vs_slippage else float("nan")
        risk_diag_avg = self._avg_diag(risk_diags)
        cov_loss_avg = self._avg_diag(cov_losses)
        benchmark_sharpe = float("nan")
        if len(benchmark_prices) > 1:
            bench_rets = np.diff(np.array(benchmark_prices)) / np.array(benchmark_prices[:-1])
            benchmark_sharpe = self._sharpe(list(bench_rets))
        if returns:
            strat_returns = np.array(returns, dtype=float).reshape(1, -1)
            ras_emp, ras_lower = ras_sharpe(strat_returns)
            ras_sharpe_emp = float(ras_emp[0]) if ras_emp.size else float("nan")
            ras_sharpe_lower = float(ras_lower[0]) if ras_lower.size else float("nan")
        else:
            ras_sharpe_emp = float("nan")
            ras_sharpe_lower = float("nan")

        metrics = BacktestMetrics(
            pnl=float(pnl),
            cum_return_pct=float(cum_return_pct),
            sharpe=float(sharpe),
            max_drawdown=float(max_dd),
            turnover=turnover,
            winrate=winrate,
            impact_vs_slippage=ivs,
            ras_sharpe=ras_sharpe_emp,
            ras_sharpe_lower=ras_sharpe_lower,
        )
        summary = {
            "pnl": pnl,
            "cum_return_pct": cum_return_pct,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "turnover": turnover,
            "winrate": winrate,
            "impact_vs_slippage": ivs,
            "ras_sharpe": ras_sharpe_emp,
            "ras_sharpe_lower": ras_sharpe_lower,
            "risk_diag_avg": risk_diag_avg,
            "cov_loss_avg": cov_loss_avg,
            "benchmark_sharpe": benchmark_sharpe,
            "ras_ic_lower": float("nan"),
        }
        self._metrics = summary

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            save_summary_json(summary, os.path.join(self.output_dir, "summary.json"))
            if equity_curve:
                with open(os.path.join(self.output_dir, "equity.csv"), "w") as f:
                    f.write("equity\n")
                    for v in equity_curve:
                        f.write(f"{v}\n")
                plot_equity(equity_curve, drawdowns, os.path.join(self.output_dir, "equity.png"))
            if risk_diag_avg:
                plot_risk_diagnostics(risk_diag_avg, os.path.join(self.output_dir, "risk_diag.png"))
            # Additional plots
            if kelly_fs or kelly_drawdowns:
                plot_series(
                    {"kelly_f": kelly_fs, "kelly_drawdown": kelly_drawdowns},
                    os.path.join(self.output_dir, "kelly.png"),
                    "Kelly & Drawdown",
                )
            if impact_estimates or slippages:
                plot_scatter(
                    impact_estimates,
                    slippages,
                    os.path.join(self.output_dir, "impact_vs_slippage.png"),
                    xlabel="Impact Estimate",
                    ylabel="Realized Slippage (bp)",
                    title="Impact vs Slippage",
                )
            if risk_series:
                with open(os.path.join(self.output_dir, "risk_series.jsonl"), "w") as f:
                    for rec in risk_series:
                        f.write(json.dumps(rec) + "\n")

        return metrics

    def _predict_with_gru(self) -> Dict[str, float]:
        """Generate expected returns using the loaded GRU forecaster."""
        if self.gru_forecaster is None:
            return {}
        expected: Dict[str, float] = {}
        lookback = self.gru_forecaster.lookback
        for sym in self.symbols:
            if self.data_engine.get_missing_bars(sym, self.timeframe):
                continue
            lr_hist = self.data_engine.return_manager.log_return_history.get(sym, [])
            vol_hist = self.data_engine.return_manager.volume_history.get(sym, [])
            if len(lr_hist) < lookback or len(vol_hist) < lookback:
                continue
            log_returns = np.array([v for _, v in lr_hist], dtype=float)
            volumes = np.array([v for _, v in vol_hist], dtype=float)
            volumes = volumes[-len(log_returns) :]
            lr_slice = log_returns[-lookback:]
            vol_slice = volumes[-lookback:]
            if lr_slice.shape[0] < lookback or vol_slice.shape[0] < lookback:
                continue
            window = np.stack([lr_slice, np.log1p(vol_slice)], axis=1)
            if not np.all(np.isfinite(window)):
                continue
            try:
                expected[sym] = self.gru_forecaster.predict_simple_return(window)
            except Exception as exc:
                logger.warning("GRU forecast failed for %s: %s", sym, exc)
        return expected

    async def _price_lookup(self, symbol: str) -> Optional[float]:
        """Async adapter around DataEngine latest price for SimBroker fills."""
        return self.data_engine.get_latest_price(symbol, self.timeframe)

    async def _warmup(self, ohlcv_data: Dict[str, Any]) -> None:
        """Ingest history prior to start to build features/risk before live walk-forward."""
        clock = sorted({ts for df in ohlcv_data.values() for ts in df.index})
        for ts in clock:
            if ts >= self.start:
                break
            self.broker.set_bar_timestamp(ts)
            for sym in self.symbols:
                df = ohlcv_data.get(sym)
                if df is not None and ts in df.index:
                    candle = df.loc[ts]
                    bar = [int(ts.timestamp() * 1000)] + candle.tolist()
                    await self.data_engine.data_fetcher.data_processor.update_tracked_candles(
                        sym, self.timeframe, bar
                    )
            self.data_engine.process_all_latest_bars(self.timeframe)
            # Keep broker prices aligned with latest bar for MTM consistency
            tracked_syms = set(self.symbols) | set(self.broker._positions.keys())  # pylint: disable=protected-access
            price_map = {s: self.data_engine.get_latest_price(s, self.timeframe) for s in tracked_syms}
            self.broker.update_last_prices(price_map)

    async def _load_history(self) -> Dict[str, Any]:
        console_log(f"Loading historical data for {len(self.symbols)} assets")
        lookback_start = _lookback_start(self.start, self.timeframe)
        tasks = [
            self.fetcher.download_ohlcv(sym, self.timeframe, lookback_start, self.end, force=False)
            for sym in self.symbols
        ]
        dfs = await asyncio.gather(*tasks)
        return {sym: df for sym, df in zip(self.symbols, dfs)}

    async def _nav(self) -> float:
        # Prefer full equity (cash + margin + unrealized PnL) if supported by broker
        if hasattr(self.broker, "equity"):
            try:
                return float(await self.broker.equity())
            except Exception:
                pass
        bal = await self.broker.get_balance()
        return float(bal["total"]["USDT"])

    @staticmethod
    def _sharpe(returns: List[float]) -> float:
        if not returns:
            return 0.0
        arr = np.array(returns, dtype=float)
        if np.std(arr, ddof=1) == 0:
            return 0.0
        return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252 * 24 * 12))

    @staticmethod
    def _max_drawdown(equity: List[float]) -> float:
        if not equity:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for x in equity:
            if x > peak:
                peak = x
            dd = (peak - x) / peak
            max_dd = max(max_dd, dd)
        return float(max_dd)

    @staticmethod
    def _avg_diag(diags: List[Dict[str, Any]]) -> Dict[str, float]:
        if not diags:
            return {}
        keys = set().union(*[d.keys() for d in diags])
        out = {}
        for k in keys:
            vals = [d.get(k) for d in diags if d.get(k) is not None]
            if vals:
                out[k] = float(np.mean(vals))
        return out


def _lookback_start(start: datetime, timeframe: str) -> datetime:
    # Fetch enough history to cover GRU lookback plus cushion
    try:
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
        elif timeframe.endswith("d"):
            hours = int(timeframe[:-1]) * 24
        else:
            hours = 1
        bars_needed = config.GRU_LOOKBACK + 50
        days = max(90, int((bars_needed * hours) / 24) + 30)
    except Exception:
        days = 90
    return start - timedelta(days=days)
