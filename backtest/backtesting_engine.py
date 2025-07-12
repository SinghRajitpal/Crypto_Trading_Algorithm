"""backtest/backtesting_engine.py

First-pass back-testing orchestrator.

Flow:
1. Load cached CSVs via HistoricalDataFetcher for every symbol/timeframe pair.
2. Merge their timestamps into a single ordered index (global clock).
3. For each bar in the clock:
   - Push latest candle into DataEngine's DataProcessor (just like live feed).
   - Update SimBroker price callback so execution uses current price.
   - Run AlgoEngine → Strategy to emit signals.
   - Pass each signal to ExecutionEngine which now targets SimBroker.
4. After loop, expose trade log and simple equity curve.

"""

from __future__ import annotations

import os, sys
# Ensure project root is on sys.path when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from datetime import UTC
from typing import Dict, List, Tuple, Any

import pandas as pd
import json  # Needed for writing summary.json when --save is enabled

from data.historical_data import HistoricalDataFetcher
from data.data_engine import DataEngine
from algorithm.algo_engine import AlgoEngine
from execution.execution_engine import ExecutionEngine
from backtest.broker import SimBroker


class BacktestingEngine:
    """Plug-and-play back-tester that re-uses live components."""

    def __init__(
        self,
        symbols: List[Tuple[str, str]],  # list of (symbol, timeframe)
        strategy,
        start,
        end,
        initial_capital: float = 10_000.0,
    ) -> None:
        self.symbols = symbols
        # Build helper mapping symbol -> list[timeframe] for quick lookup
        self._symbol_tfs: Dict[str, List[str]] = {}
        for sym, tf in symbols:
            self._symbol_tfs.setdefault(sym, []).append(tf)
        self.strategy = strategy
        self.start = start
        self.end = end

        # Simulated broker
        self.broker = SimBroker(initial_capital)

        # Data loader
        self.fetcher = HistoricalDataFetcher()

        # DataEngine with small maxlen (strategy dictates requirements)
        self.data_engine = DataEngine(binance_client=self.broker, max_candles=500)

        # Algo & Execution engines wired to simulated broker
        self.algo_engine = AlgoEngine(self.data_engine)
        self.execution_engine = ExecutionEngine(binance_client=self.broker, total_capital=initial_capital)

        # Associate strategy with algo_engine
        self.strategy.set_algo_engine(self.algo_engine)

        # Storage
        self._raw_data: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._funding_data: Dict[str, pd.Series] = {}

    async def load_data(self):
        """Fetch or read cached data for all symbol/timeframes."""
        tasks = []
        for sym, tf in self.symbols:
            tasks.append(self.fetcher.download_ohlcv(sym, tf, self.start, self.end))
        # Also schedule funding-rate downloads (once per symbol)
        sym_set = {sym for sym, _ in self.symbols}
        for sym in sym_set:
            tasks.append(self.fetcher.fetch_funding_rate(sym, self.start, self.end))
        dfs = await asyncio.gather(*tasks)
        # First len(self.symbols) items correspond to OHLCV results
        ohlcv_results = dfs[: len(self.symbols)]
        funding_results = dfs[len(self.symbols):]

        for (sym, tf), df in zip(self.symbols, ohlcv_results):
            self._raw_data[(sym, tf)] = df

        for sym, series in zip(sym_set, funding_results):
            self._funding_data[sym] = series

    async def _price_lookup(self, symbol: str) -> float:
        """Return latest close price for *symbol* (any available timeframe)."""
        # Try preferred timeframes in order they were registered
        for tf in self._symbol_tfs.get(symbol, []):
            price = self.data_engine.get_latest_price(symbol, tf)
            if price is not None:
                return price
        # If none available yet, return 0 to indicate unavailable
        return 0.0

    async def run(self):
        """Execute the backtest and return trade log & simple metrics."""
        await self.load_data()

        # Build global clock (timestamps) 
        all_ts = set()
        for df in self._raw_data.values():
            all_ts.update(df.index)
        clock = sorted(all_ts)

        # Attach price callback
        self.broker.set_price_callback(self._price_lookup)

        # Main loop 
        for ts in clock:
            # Tell the broker which simulated time we are at for proper logging
            self.broker.set_bar_timestamp(ts)
            # For each symbol/timeframe add rows up to current ts
            for (sym, tf), df in self._raw_data.items():
                if ts in df.index:
                    candle = df.loc[ts]
                    await self.data_engine.data_fetcher.data_processor.update_tracked_candles(
                        sym, tf, [int(ts.timestamp()*1000)] + candle.tolist()
                    )

            # Skip trade execution until the requested *start* date – previous
            # bars are used only to warm-up indicators so that the strategy
            # has sufficient history before trading begins.
            if ts < self.start:
                continue

            # Evaluate stop-loss / take-profit for open positions
            for sym, pos in list(self.broker._positions.items()):  # type: ignore
                current_price = await self._price_lookup(sym)
                sl = pos.get("stop_loss")
                tp = pos.get("take_profit")

                trigger = False
                if pos["contracts"] > 0:  # long
                    if sl is not None and current_price <= sl:
                        trigger = True
                    elif tp is not None and current_price >= tp:
                        trigger = True
                else:  # short
                    if sl is not None and current_price >= sl:
                        trigger = True
                    elif tp is not None and current_price <= tp:
                        trigger = True

                if trigger:
                    # Close via broker and immediately free reserved capital
                    await self.broker.close_position(sym)
                    try:
                        # Release *all* allocation reserved for this symbol
                        self.execution_engine.portfolio_manager.release_allocation(sym)
                    except KeyError:
                        # Symbol might not have an entry if it was never allocated
                        pass

            # Process signals for each symbol/timeframe
            for sym, tf in self.symbols:
                signal = await self.algo_engine.process_signals(sym, tf, self.strategy)
                if not signal:
                    continue

                # Attach latest price so ExecutionEngine has it
                current_price = await self._price_lookup(sym)
                signal.metadata["price"] = current_price

                # Run risk validation for open signals (mirrors TradingAlgorithm)
                if signal.action == "open":
                    risk_result = await self.execution_engine.validate_signal(signal, current_price)
                    signal.metadata["risk_valid"] = risk_result.get("valid", False)
                    signal.metadata["risk_reason"] = risk_result.get("reason", "Unknown reason")

                await self.execution_engine.process_signal(signal)

            # Check funding timestamps (every 8h at 00:00,08:00,16:00 UTC)
            if ts.hour % 8 == 0 and ts.minute == 0:
                for sym in self._symbol_tfs.keys():
                    series = self._funding_data.get(sym)
                    if series is None or ts not in series.index:
                        continue
                    rate = series.loc[ts]
                    await self.broker.apply_funding(sym, rate)

        # ------------------------------------------------------------------
        # End-of-test cleanup – close any remaining open positions *and*
        # release their reserved allocations so future back-tests start from
        # a clean slate.
        # ------------------------------------------------------------------
        closed = await self.broker.close_all_positions()

        for sym in closed.keys():
            try:
                self.execution_engine.portfolio_manager.release_allocation(sym)
            except KeyError:
                pass

        # Close fetcher exchange connection cleanly
        await self.fetcher.close()

        trades = self.broker.trade_log()
        final_equity = await self.broker.equity()

        # Metrics
        try:
            from backtest.metrics import Metrics
            metrics = Metrics(trades, self.broker.initial_capital)
            summary = metrics.summary()
            equity_curve = metrics.equity_curve()
        except Exception as e:
            print(f"[Backtesting] Metrics error: {e}")
            summary = {}
            equity_curve = None
        return {
            "trades": trades,
            "final_cash": final_equity,  # kept key name for CLI compatibility
            "trade_count": int((trades["type"] == "close").sum()),
            "summary": summary,
            "equity_curve": equity_curve,
        }

# ---------------------------------------------------------------------------
# CLI driver – allows `python backtest/backtesting_engine.py` quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from datetime import datetime, timedelta, UTC
    from algorithm.strategies.ma_crossover import MACrossoverStrategy

    parser = argparse.ArgumentParser(description="Run a quick back-test.")
    parser.add_argument("symbols", nargs="?", default="", help="Comma separated trading pairs. Leave empty or use 'ALL' to back-test every configured coin")
    parser.add_argument("--tf", default="5m", help="Timeframe, default 5m")
    parser.add_argument("--days", type=int, default=3, help="Days of history, default 3")
    # Optional explicit date range overrides --days. Format: DD/MM/YYYY
    parser.add_argument("--start", type=str, default=None, help="Start date (DD/MM/YYYY)")
    parser.add_argument("--end", type=str, default=None, help="End date (DD/MM/YYYY). Defaults to today if omitted")
    # Optional flag to persist stats & plots -------------------------------------------------
    parser.add_argument("--save", action="store_true", help="Save back-test stats and plots to disk")
    args = parser.parse_args()

    if not args.symbols.strip() or args.symbols.strip().upper() == "ALL":
        import config
        symbols = [(sym.upper(), args.tf) for sym, _ in config.symbols]
        print(f"[Backtesting] Running ALL configured symbols: {', '.join([s for s, _ in symbols])}")
    else:
        symbols = [(sym.strip().upper(), args.tf) for sym in args.symbols.split(",") if sym.strip()]

    # ------------------------------------------------------------------
    # Determine back-test date range ------------------------------------
    # Priority: explicit --start / --end > --days fallback
    # ------------------------------------------------------------------
    date_format = "%d/%m/%Y"

    if args.start:
        try:
            start_dt = datetime.strptime(args.start, date_format).replace(tzinfo=UTC)
        except ValueError:
            raise SystemExit(f"[Backtesting] Invalid --start date format. Expected DD/MM/YYYY, got {args.start}")

        # Handle --end; default to today UTC if not provided
        if args.end:
            try:
                end_dt = datetime.strptime(args.end, date_format).replace(tzinfo=UTC)
            except ValueError:
                raise SystemExit(f"[Backtesting] Invalid --end date format. Expected DD/MM/YYYY, got {args.end}")
        else:
            end_dt = datetime.now(UTC)

        if end_dt <= start_dt:
            raise SystemExit("[Backtesting] --end date must be after --start date")
    else:
        # Fallback to --days offset
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=args.days)

    strategy = MACrossoverStrategy()

    engine = BacktestingEngine(
        symbols=symbols,
        strategy=strategy,
        start=start_dt,
        end=end_dt,
        initial_capital=10_000,
    )

    # ------------------------------------------------------------------
    # Run the back-test --------------------------------------------------
    # ------------------------------------------------------------------
    result = asyncio.run(engine.run())

    # ------------------------------------------------------------------
    # Prepare optional output directory if --save passed -----------------
    # ------------------------------------------------------------------
    save_dir = None
    if args.save:
        from datetime import datetime
        ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        # Nest results by strategy id so multiple strategies stay separate
        base_results_dir = os.path.join(os.path.dirname(__file__), "results", strategy.strategy_id)
        save_dir = os.path.join(base_results_dir, ts_str)

        # Sub-folders for figures
        indiv_dir = os.path.join(save_dir, "individual_asset_performance")
        port_dir = os.path.join(save_dir, "portfolio_performance")

        # Create directories
        for d in [indiv_dir, port_dir]:
            os.makedirs(d, exist_ok=True)

        print(f"[Backtesting] Saving report to {save_dir}")

    # Visualisation / stats -------------------------------------------------------
    try:
        import vectorbt as vbt  # noqa: F401
        from backtest.vectorbt_adapter import (
            load_close_dataframe,
            portfolio_from_trades_multi,
            portfolio_from_trades,
        )

        # ------------------------------------------------------------------
        # Load close-price history for *all* requested symbols so that the
        # resulting vectorbt Portfolio represents the whole portfolio rather
        # than just the first asset.
        # ------------------------------------------------------------------

        close_df = load_close_dataframe(symbols, start=start_dt, end=end_dt)

        pf = portfolio_from_trades_multi(result["trades"], close_df, 10_000)

        print("\nPORTFOLIO STATISTICS (all assets):")

        # ------------------------------------------------------------------
        # NEW: Per-asset statistics & plots when testing multiple assets
        # ------------------------------------------------------------------
        if close_df.shape[1] > 1:
            import pandas as pd

            per_asset_stats: dict[str, pd.Series] = {}

            for asset in close_df.columns:
                # Filter trade-log for this asset only
                asset_trades = result["trades"].loc[result["trades"]["symbol"] == asset]
                if asset_trades.empty:
                    print(f"[Backtesting] No trades for {asset} – skipping stats & plot")
                    continue

                try:
                    # Build standalone Portfolio for this asset using full initial capital (10k)
                    # to evaluate its intrinsic performance irrespective of portfolio allocation.
                    asset_pf = portfolio_from_trades(asset_trades, close_df[asset], engine.broker.initial_capital)

                    per_asset_stats[asset] = asset_pf.stats()

                    # Quick performance figure (value & drawdowns)
                    subplots_avail = set(asset_pf.subplots.keys())
                    subplots_used = [s for s in ["value", "drawdowns"] if s in subplots_avail]
                    asset_fig = asset_pf.plots(subplots=subplots_used)
                    asset_fig.update_layout(title_text=f"{asset} Performance")
                    if args.save and save_dir is not None:
                        asset_fig.write_html(os.path.join(indiv_dir, f"{asset}.html"))
                    else:
                        asset_fig.show()
                except Exception as e:
                    print(f"[Vectorbt] Could not analyse asset {asset}: {e}")

            if per_asset_stats:
                print("\nINDIVIDUAL ASSET STATISTICS:")
                print(pd.DataFrame(per_asset_stats))

        # ------------------------------------------------------------------
        # Stats – for multi-asset portfolios we ask vectorbt to aggregate
        # across the first-level (asset) dimension by passing ``group_by=True``.
        # For single-asset, the default behaviour is fine.
        # ------------------------------------------------------------------

        try:
            stats_ser = pf.stats(group_by=True) if close_df.shape[1] > 1 else pf.stats()
        except Exception:
            stats_ser = pf.stats()
        print(stats_ser)

        # Persist statistics if requested --------------------------------
        if args.save and save_dir is not None:
            try:
                # Portfolio-level stats
                stats_ser.to_csv(os.path.join(port_dir, "portfolio_stats.csv"))

                # Per-asset stats DataFrame if available
                if "per_asset_stats" in locals() and per_asset_stats:
                    import pandas as pd
                    pd.DataFrame(per_asset_stats).to_csv(os.path.join(indiv_dir, "per_asset_stats.csv"))

                # Trade log at top-level of this back-test folder
                result["trades"].to_csv(os.path.join(save_dir, "trade_log.csv"))

                # Rebuild concise metrics based on vectorbt stats for accuracy
                # Pull common metrics from vectorbt stats – fall back to SimBroker values
                metrics_dict = {
                    "final_equity": float(pf.value().iloc[-1]) if "pf" in locals() else result["final_cash"],
                    "total_return_pct": stats_ser.get("Total Return [%]", result["summary"].get("total_return_pct")),
                    "max_drawdown_pct": stats_ser.get("Max Drawdown [%]", result["summary"].get("max_drawdown_pct")),
                    "sharpe": stats_ser.get("Sharpe Ratio", result["summary"].get("sharpe")),
                    # Prefer vectorbt 'Total Trades' if available; otherwise SimBroker count of closed trades
                    "trade_count": int(stats_ser.get("Total Trades", result["trade_count"])),
                }

                summary_payload = {
                    "metrics": metrics_dict,                  # concise metrics derived from portfolio
                    "portfolio_stats": stats_ser.to_dict(),   # full vectorbt statistics
                    "timespan": {
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                        "days": (end_dt - start_dt).days
                    }
                }

                # Include per‐asset statistics when available
                if "per_asset_stats" in locals() and per_asset_stats:
                    summary_payload["per_asset_stats"] = {k: v.to_dict() for k, v in per_asset_stats.items()}

                with open(os.path.join(save_dir, "summary.json"), "w") as fh:
                    json.dump(summary_payload, fh, indent=2, default=str)
            except Exception as e:
                print(f"[Backtesting] Could not save results: {e}")

        wanted = [
            "orders",      # individual order markers per asset
            "trades",      # entry/exit markers
            "value",       # cumulative equity curve (cum-returns)
            "assets",      # per-asset market value
            "drawdowns",   # under-water curve
        ]

        # Only keep names that exist in this version of vectorbt
        available = set(pf.subplots.keys())
        subplots = [s for s in wanted if s in available]

        multi_asset = close_df.shape[1] > 1

        # For multi-asset portfolios vectorbt panels like 'orders', 'trades',
        # and even 'value' expect a single column.  The recommended approach is
        # to plot with ``group_by='group'`` which aggregates all columns into
        # one virtual group (the full portfolio).  We therefore:
        #   • Drop orders / trades panels (they are per-asset and noisy).
        #   • Plot using group_by='group' so that 'value' and others work.

        if multi_asset:
            # Aggregate across assets using vectorbt's built-in grouping
            group_by_opt = True  # use first level of column MultiIndex
            # Detailed per-order panels and 'assets' don't support grouped data well
            subplots = [s for s in subplots if s not in {"orders", "trades", "assets"}]
        else:
            # Single-asset plotting: keep all subplots and don't group columns
            group_by_opt = None

        # Build figure – show or persist
        fig = pf.plots(subplots=subplots, group_by=group_by_opt)

        if args.save and save_dir is not None:
            fig.write_html(os.path.join(port_dir, "portfolio.html"))
        else:
            fig.show()  # Interactive display
    except ImportError:
        print("[Vectorbt] vectorbt not installed – install with 'pip install vectorbt' to view plots and stats")
    except Exception as e:
        print(f"[Vectorbt] Could not build or plot portfolio: {e}")

    # ------------------------------------------------------------------
    # Consistent final-equity display ----------------------------------
    # Use vectorbt Portfolio end value when available so that the printed
    # figure matches the stats table (avoids confusion with SimBroker
    # cash balance which ignores leverage effects).
    # ------------------------------------------------------------------

    final_equity_vbt = None
    try:
        final_equity_vbt = float(pf.value().iloc[-1])
    except Exception:
        # Fallback to SimBroker equity if vectorbt failed
        final_equity_vbt = result["final_cash"]

    print("\nBack-test complete")
    print(f"Trades executed : {result['trade_count']}")
    print(f"Final equity    : {final_equity_vbt:.2f} USDT")
    if result['trade_count']:
        print("Last trades:\n", result['trades'].tail())

    if args.save and save_dir is not None:
        print(f"[Backtesting] Results saved to {save_dir}")

    # Finished – plots shown above. No extra debug prints.
