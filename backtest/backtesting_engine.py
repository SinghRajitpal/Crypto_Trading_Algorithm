from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config
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
    plot_equity_comparison,
    plot_risk_diagnostics,
    plot_series,
    save_summary_json,
)

logger = get_logger(__name__)


@dataclass
class BacktestMetrics:
    pnl: float
    cum_return_pct: float
    sharpe: float
    volatility: float
    max_drawdown: float
    avg_drawdown: float
    turnover: float
    turnover_sum: float
    winrate: float
    avg_win: float
    avg_loss: float
    fees: float
    ras_sharpe: float = float("nan")
    ras_sharpe_lower: float = float("nan")
    benchmark_sharpe: float = float("nan")
    benchmark_cum_return_pct: float = float("nan")
    benchmark_pnl: float = float("nan")
    benchmark_volatility: float = float("nan")
    benchmark_max_drawdown: float = float("nan")
    benchmark_avg_drawdown: float = float("nan")
    benchmark_avg_win: float = float("nan")
    benchmark_avg_loss: float = float("nan")


class WalkForwardBacktester:
    """Layer B walk-forward backtest using live components with per-asset retraining."""

    def __init__(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        predictions_paths: List[str],
        initial_capital: float = 10_000.0,
        output_dir: Optional[str] = None,
        benchmark_symbol: Optional[str] = None,
        benchmark_dir: Optional[str] = None,
    ) -> None:
        self.symbols = symbols
        self.start = start
        self.end = end
        self.predictions_paths = predictions_paths
        self.initial_capital = initial_capital
        self.output_dir = output_dir
        self.benchmark_symbol = benchmark_symbol or getattr(config, "BENCHMARK_SYMBOL", None)
        self.benchmark_dir = benchmark_dir
        self.timeframe = config.GRU_TIMEFRAME
        self._pred_map: Dict[int, Dict[str, float]] = {}
        self._pred_df: Optional[pd.DataFrame] = None

        self.broker = SimBroker(initial_capital)
        self.data_engine = DataEngine(binance_client=self.broker, max_candles=2000)
        self.data_engine.primary_timeframe = self.timeframe
        self.data_engine.data_fetcher.symbol_timeframes = [(s, self.timeframe) for s in symbols]
        self.execution_engine = ExecutionEngine(binance_client=self.broker, total_capital=initial_capital)
        self.fetcher = HistoricalDataFetcher(demo=False)
        self._metrics: Dict[str, Any] = {}
        # Use latest processed price for simulated fills
        self.broker.set_price_callback(self._price_lookup)

    async def run(self) -> BacktestMetrics:
        try:
            ohlcv_data, benchmark_df = await self._load_history(self.benchmark_symbol)
            await self._warmup(ohlcv_data)
            self._pred_map = self._load_prediction_feed(self.predictions_paths, self.start, self.end)
            self._validate_predictions(self.start, self.end)
            clock = sorted({ts for df in ohlcv_data.values() for ts in df.index})

            equity_curve: List[float] = []
            returns: List[float] = []
            turnovers: List[float] = []
            fees_list: List[float] = []
            wins = 0
            trades = 0
            risk_diags: List[Dict[str, Any]] = []
            cov_losses: List[Dict[str, Any]] = []
            drawdowns: List[float] = []
            peak = self.initial_capital
            kelly_fs: List[float] = []
            kelly_drawdowns: List[float] = []
            risk_series: List[Dict[str, Any]] = []
            slippages: List[float] = []
            benchmark_equity_curve: List[float] = []
            benchmark_returns: List[float] = []
            benchmark_drawdowns: List[float] = []
            benchmark_units: Optional[float] = None
            benchmark_peak = self.initial_capital
            benchmark_last_price: Optional[float] = None

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

                expected_returns = self._expected_returns(ts)
                logger.debug(
                    "Bar %s | forecasts=%d | universe=%s",
                    ts.isoformat(),
                    len(expected_returns),
                    list(expected_returns.keys()),
                )

                bar_turnover = 0.0
                bar_fees = 0.0
                if expected_returns:
                    diagnostics = {"model": "gru_layerA_feed"}
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
                        # still record equity later
                        pass
                    else:
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
                        bar_turnover = result.get("turnover", 0.0) or 0.0
                        bar_fees = result.get("expected_fee", 0.0) or 0.0
                        trades += len(result.get("orders", []))
                        if result.get("risk_diag"):
                            risk_diags.append(result.get("risk_diag"))
                        if result.get("risk_cov_loss"):
                            cov_losses.append(result.get("risk_cov_loss"))
                        # Capture time-series risk/costs
                        risk_series.append(
                            {
                                "ts": forecast.timestamp,
                                "risk_diag": result.get("risk_diag"),
                                "risk_cov_loss": result.get("risk_cov_loss"),
                                "turnover": result.get("turnover"),
                                "expected_cost": result.get("expected_cost"),
                                "expected_cost_bp": result.get("expected_cost_bp"),
                                "expected_fee": result.get("expected_fee"),
                                "expected_slippage": result.get("expected_slippage"),
                            }
                        )
                        if result.get("kelly_f") is not None:
                            kelly_fs.append(result.get("kelly_f"))
                        if result.get("kelly_drawdown") is not None:
                            kelly_drawdowns.append(result.get("kelly_drawdown"))
                        if result.get("realized_slippage_bp") is not None:
                            slippages.append(result.get("realized_slippage_bp"))

                nav_new = await self._nav()
                equity_curve.append(nav_new)
                peak = max(peak, nav_new)
                dd = (peak - nav_new) / peak if peak > 0 else 0.0
                drawdowns.append(dd)
                # Benchmark: buy-and-hold BTC (or configured symbol)
                bench_price = None
                if self.benchmark_symbol:
                    if self.benchmark_symbol in self.symbols:
                        bench_price = self.data_engine.get_latest_price(self.benchmark_symbol, self.timeframe)
                    elif benchmark_df is not None and ts in benchmark_df.index:
                        bench_price = float(benchmark_df.loc[ts]["close"])
                    if bench_price is None and benchmark_last_price is not None:
                        bench_price = benchmark_last_price

                if bench_price is not None and bench_price > 0:
                    benchmark_last_price = bench_price
                    if benchmark_units is None:
                        benchmark_units = self.initial_capital / bench_price
                    bench_nav = benchmark_units * bench_price if benchmark_units is not None else None
                    if bench_nav is not None:
                        benchmark_equity_curve.append(bench_nav)
                        benchmark_peak = max(benchmark_peak, bench_nav)
                        bench_dd = (benchmark_peak - bench_nav) / benchmark_peak if benchmark_peak > 0 else 0.0
                        benchmark_drawdowns.append(bench_dd)
                        if len(benchmark_equity_curve) > 1:
                            bench_prev = benchmark_equity_curve[-2]
                            if bench_prev != 0:
                                benchmark_returns.append((bench_nav - bench_prev) / bench_prev)

                if len(equity_curve) > 1:
                    r = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                    returns.append(r)
                    if r > 0 and expected_returns:
                        wins += 1
                turnovers.append(bar_turnover)
                fees_list.append(bar_fees)

            pnl = equity_curve[-1] - self.initial_capital if equity_curve else 0.0
            cum_return_pct = (equity_curve[-1] / self.initial_capital - 1) * 100 if equity_curve else 0.0
            bars_per_day = self._bars_per_day(self.timeframe)
            sharpe = self._sharpe(returns, bars_per_day)
            vol = self._volatility(returns, bars_per_day)
            max_dd = self._max_drawdown(equity_curve)
            avg_dd = float(np.mean(drawdowns)) if drawdowns else 0.0
            turnover = float(np.mean(turnovers)) if turnovers else 0.0
            turnover_sum = float(np.sum(turnovers)) if turnovers else 0.0
            winrate = wins / max(1, trades)
            risk_diag_avg = self._avg_diag(risk_diags)
            cov_loss_avg = self._avg_diag(cov_losses)
            avg_win, avg_loss = self._avg_win_loss(returns)
            fees_total = float(np.sum(fees_list)) if fees_list else 0.0
            benchmark_pnl = benchmark_equity_curve[-1] - self.initial_capital if benchmark_equity_curve else 0.0
            benchmark_cum_return_pct = (
                (benchmark_equity_curve[-1] / self.initial_capital - 1) * 100 if benchmark_equity_curve else 0.0
            )
            benchmark_sharpe = self._sharpe(benchmark_returns, bars_per_day)
            benchmark_vol = self._volatility(benchmark_returns, bars_per_day)
            benchmark_max_dd = self._max_drawdown(benchmark_equity_curve)
            benchmark_avg_dd = float(np.mean(benchmark_drawdowns)) if benchmark_drawdowns else 0.0
            benchmark_avg_win, benchmark_avg_loss = self._avg_win_loss(benchmark_returns)
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
                volatility=float(vol),
                max_drawdown=float(max_dd),
                avg_drawdown=float(avg_dd),
                turnover=turnover,
                turnover_sum=turnover_sum,
                winrate=winrate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                fees=fees_total,
                ras_sharpe=ras_sharpe_emp,
                ras_sharpe_lower=ras_sharpe_lower,
                benchmark_sharpe=benchmark_sharpe,
                benchmark_cum_return_pct=benchmark_cum_return_pct,
                benchmark_pnl=benchmark_pnl,
                benchmark_volatility=benchmark_vol,
                benchmark_max_drawdown=benchmark_max_dd,
                benchmark_avg_drawdown=benchmark_avg_dd,
                benchmark_avg_win=benchmark_avg_win,
                benchmark_avg_loss=benchmark_avg_loss,
            )
            summary = {
                "pnl": pnl,
                "cum_return_pct": cum_return_pct,
                "sharpe": sharpe,
                "volatility": vol,
                "max_drawdown": max_dd,
                "avg_drawdown": avg_dd,
                "turnover": turnover,
                "turnover_sum": turnover_sum,
                "winrate": winrate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "fees": fees_total,
                "ras_sharpe": ras_sharpe_emp,
                "ras_sharpe_lower": ras_sharpe_lower,
                "risk_diag_avg": risk_diag_avg,
                "cov_loss_avg": cov_loss_avg,
                "benchmark_sharpe": benchmark_sharpe,
                "ras_ic_lower": float("nan"),
            }
            benchmark_summary = {
                "pnl": benchmark_pnl,
                "cum_return_pct": benchmark_cum_return_pct,
                "sharpe": benchmark_sharpe,
                "volatility": benchmark_vol,
                "max_drawdown": benchmark_max_dd,
                "avg_drawdown": benchmark_avg_dd,
                "avg_win": benchmark_avg_win,
                "avg_loss": benchmark_avg_loss,
            }
            summary.update({f"benchmark_{k}": v for k, v in benchmark_summary.items()})
            self._metrics = summary

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                save_summary_json(summary, os.path.join(self.output_dir, "summary.json"))
                # Also write metrics.csv for easy ingestion
                with open(os.path.join(self.output_dir, "metrics.csv"), "w") as f:
                    f.write("metric,value\n")
                    for k, v in summary.items():
                            f.write(f"{k},{v}\n")
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
                if risk_series:
                    with open(os.path.join(self.output_dir, "risk_series.jsonl"), "w") as f:
                        for rec in risk_series:
                            f.write(json.dumps(rec) + "\n")
                if self.benchmark_dir and benchmark_equity_curve:
                    os.makedirs(self.benchmark_dir, exist_ok=True)
                    save_summary_json(benchmark_summary, os.path.join(self.benchmark_dir, "summary.json"))
                    with open(os.path.join(self.benchmark_dir, "metrics.csv"), "w") as f:
                        f.write("metric,value\n")
                        for k, v in benchmark_summary.items():
                            f.write(f"{k},{v}\n")
                    with open(os.path.join(self.benchmark_dir, "equity.csv"), "w") as f:
                        f.write("equity\n")
                        for v in benchmark_equity_curve:
                            f.write(f"{v}\n")
                    plot_equity(benchmark_equity_curve, benchmark_drawdowns, os.path.join(self.benchmark_dir, "equity.png"))
                    plot_equity_comparison(
                        equity_curve, benchmark_equity_curve, os.path.join(self.benchmark_dir, "equity_comparison.png")
                    )
            return metrics
        finally:
            try:
                await self.fetcher.close()
            except Exception:
                pass

    def _expected_returns(self, ts: datetime) -> Dict[str, float]:
        """Read pre-computed expected returns for the current bar."""
        ts_ms = int(ts.timestamp() * 1000)
        raw = self._pred_map.get(ts_ms, {})
        if not raw:
            return {}
        expected: Dict[str, float] = {}
        for sym, log_ret in raw.items():
            if sym not in self.symbols:
                continue
            # Convert predicted log-return to simple return for portfolio layer
            expected[sym] = float(np.exp(log_ret) - 1.0)
        return expected

    def _load_prediction_feed(self, paths: List[str], start: datetime, end: datetime) -> Dict[int, Dict[str, float]]:
        """Load Layer A predictions into an in-memory lookup keyed by timestamp."""
        if not paths:
            raise ValueError("predictions_paths is required for Layer B backtest.")
        frames: List[pd.DataFrame] = []
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Predictions file not found: {path}")
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if "timestamp_ms" not in df.columns:
                if "timestamp" not in df.columns:
                    raise ValueError(f"Predictions file missing timestamp columns: {path}")
                df["timestamp_ms"] = (df["timestamp"].astype("int64") // 1_000_000).astype("int64")
            frames.append(df[["timestamp_ms", "symbol", "y_pred_logret"]])
        if not frames:
            return {}
        df_all = pd.concat(frames, ignore_index=True)
        df_all["symbol"] = df_all["symbol"].astype(str).str.upper()
        df_all = df_all.dropna(subset=["timestamp_ms", "symbol", "y_pred_logret"])
        df_all.sort_values("timestamp_ms", inplace=True)
        # Filter to backtest window to reduce noise/memory
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        df_all = df_all[(df_all["timestamp_ms"] >= start_ms) & (df_all["timestamp_ms"] <= end_ms)]
        pred_map: Dict[int, Dict[str, float]] = {}
        for row in df_all.itertuples(index=False):
            ts_ms = int(row.timestamp_ms)
            sym = row.symbol
            pred_map.setdefault(ts_ms, {})[sym] = float(row.y_pred_logret)
        self._pred_df = df_all
        return pred_map

    def _validate_predictions(self, start: datetime, end: datetime) -> None:
        """Sanity checks on prediction coverage and symbols."""
        if self._pred_df is None or self._pred_df.empty:
            raise ValueError("No predictions loaded for Layer B backtest.")
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        min_ts = int(self._pred_df["timestamp_ms"].min())
        max_ts = int(self._pred_df["timestamp_ms"].max())
        if min_ts > start_ms or max_ts < end_ms:
            logger.warning(
                "Prediction window [%s, %s] does not fully cover backtest window [%s, %s].",
                pd.to_datetime(min_ts, unit="ms", utc=True).isoformat(),
                pd.to_datetime(max_ts, unit="ms", utc=True).isoformat(),
                start.isoformat(),
                end.isoformat(),
            )
        # Warn on symbols with no predictions
        preds_symbols = set(self._pred_df["symbol"].unique())
        missing = [s for s in self.symbols if s.upper() not in preds_symbols]
        if missing:
            logger.warning("No predictions found for symbols: %s", missing)

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

    async def _load_history(self, benchmark_symbol: Optional[str] = None) -> tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        console_log(f"Loading historical data for {len(self.symbols)} assets")
        lookback_start = _lookback_start(self.start, self.timeframe)
        tasks = [
            self.fetcher.download_ohlcv(sym, self.timeframe, lookback_start, self.end, force=False)
            for sym in self.symbols
        ]
        dfs = await asyncio.gather(*tasks)
        data_map = {sym: df for sym, df in zip(self.symbols, dfs)}
        benchmark_df: Optional[pd.DataFrame] = None
        if benchmark_symbol:
            if benchmark_symbol in data_map:
                benchmark_df = data_map[benchmark_symbol]
            else:
                benchmark_df = await self.fetcher.download_ohlcv(
                    benchmark_symbol, self.timeframe, lookback_start, self.end, force=False
                )
        return data_map, benchmark_df

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
    def _sharpe(returns: List[float], bars_per_day: float) -> float:
        if not returns:
            return 0.0
        arr = np.array(returns, dtype=float)
        if np.std(arr, ddof=1) == 0:
            return 0.0
        annual_factor = np.sqrt(max(1.0, bars_per_day * 252))
        return float(np.mean(arr) / np.std(arr, ddof=1) * annual_factor)

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
    def _volatility(returns: List[float], bars_per_day: float) -> float:
        if not returns:
            return 0.0
        arr = np.array(returns, dtype=float)
        if arr.size < 2:
            return 0.0
        annual_factor = np.sqrt(max(1.0, bars_per_day * 252))
        return float(np.std(arr, ddof=1) * annual_factor)

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

    @staticmethod
    def _bars_per_day(timeframe: str) -> float:
        try:
            if timeframe.endswith("h"):
                hours = int(timeframe[:-1])
                return max(1.0, 24.0 / hours)
            if timeframe.endswith("d"):
                days = int(timeframe[:-1])
                return max(1.0, 1.0 / days)
        except Exception:
            pass
        return 1.0

    @staticmethod
    def _avg_win_loss(returns: List[float]) -> tuple[float, float]:
        if not returns:
            return 0.0, 0.0
        arr = np.array(returns, dtype=float)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        avg_win = float(np.mean(wins)) if wins.size else 0.0
        avg_loss = float(np.mean(np.abs(losses))) if losses.size else 0.0
        return avg_win, avg_loss


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
