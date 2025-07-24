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
                    # Close via broker with slippage and immediately free reserved capital
                    await self.broker.close_position(sym, slippage_bp=3.0)
                    try:
                        # Release *all* allocation reserved for this symbol
                        allocation_amount = self.execution_engine.portfolio_manager.get_allocated_capital(sym)
                        if allocation_amount > 0:
                            self.execution_engine.portfolio_manager.release_allocation(sym, allocation_amount)
                    except (KeyError, AttributeError):
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
                # Get the current allocation amount before releasing
                allocation_amount = self.execution_engine.portfolio_manager.get_allocated_capital(sym)
                if allocation_amount > 0:
                    self.execution_engine.portfolio_manager.release_allocation(sym, allocation_amount)
            except (KeyError, AttributeError):
                pass

        # Close fetcher exchange connection cleanly
        await self.fetcher.close()

        trades = self.broker.trade_log()
        final_equity = await self.broker.equity()

        # Basic trade statistics only (detailed metrics will be handled by QuantStats-Lumi)
        # Handle case where trades DataFrame might be empty or have no 'type' column
        trade_count = 0
        if not trades.empty and "type" in trades.columns:
            trade_count = int((trades["type"] == "close").sum())
        
        return {
            "trades": trades,
            "final_cash": final_equity,  # kept key name for CLI compatibility
            "trade_count": trade_count,
        }

# ---------------------------------------------------------------------------
# CLI driver – allows `python backtest/backtesting_engine.py` quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from datetime import datetime, timedelta, UTC
    from algorithm.strategies.ma_crossover import MACrossoverStrategy

    parser = argparse.ArgumentParser(description="Run a quick back-test.")
    parser.add_argument("--symbols", default="", help="Comma separated trading pairs. Leave empty or use 'ALL' to back-test every configured coin")
    parser.add_argument("--strategy", default="ma_crossover", choices=["ma_crossover"], help="Trading strategy to use (default: ma_crossover)")
    parser.add_argument("--tf", default="5m", help="Timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d, etc. (default: 5m)")
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

    # ------------------------------------------------------------------
    # Strategy initialization based on user selection
    # ------------------------------------------------------------------
    if args.strategy == "ma_crossover":
        strategy = MACrossoverStrategy()
    else:
        # Future strategies can be added here
        raise ValueError(f"Unknown strategy: {args.strategy}")

    print(f"[Backtesting] Strategy: {strategy.strategy_id}")

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

    # Visualization / stats with quantstats-lumi only -------------------------
    try:
        from backtest.visualizer import QuantStatsVisualizer
        
        print("\nPORTFOLIO STATISTICS (QuantStats-Lumi):")
        
        # Extract just symbol names and timeframe
        symbol_names = [sym for sym, _ in symbols]
        primary_timeframe = symbols[0][1] if symbols else "5m"
        
        # Initialize visualizer
        visualizer = QuantStatsVisualizer(initial_capital=10_000.0)
        
        # Generate portfolio analysis
        portfolio_returns, portfolio_metrics = visualizer.generate_portfolio_report(
            trades_df=result["trades"],
            symbols=symbol_names,
            start_date=start_dt,
            end_date=end_dt,
            timeframe=primary_timeframe,
            benchmark_symbol=symbol_names[0] if symbol_names else None  # Use first symbol as benchmark
        )
        
        # Generate individual asset analysis
        asset_results = {}
        for symbol in symbol_names:
            try:
                asset_returns, asset_metrics = visualizer.generate_asset_report(
                    trades_df=result["trades"],
                    symbol=symbol,
                    start_date=start_dt,
                    end_date=end_dt,
                    timeframe=primary_timeframe,
                    benchmark_symbol=symbol  # Use asset itself as benchmark
                )
                
                if asset_returns is not None and asset_metrics:
                    asset_results[symbol] = (asset_returns, asset_metrics)
                
            except Exception as e:
                print(f"[QuantStats-Lumi] Could not analyze asset {symbol}: {e}")
        
        # Print comprehensive metrics summary using QuantStats-Lumi
        visualizer.print_summary(portfolio_metrics, {k: v[1] for k, v in asset_results.items()})
        
        # Additional trade information
        final_equity_display = portfolio_metrics.get("Final Equity", result["final_cash"])
        print(f"\n📊 TRADE SUMMARY")
        print(f"Trades executed: {result['trade_count']}")
        print(f"Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        if result['trade_count'] > 0:
            print(f"\n📋 RECENT TRADES:")
            recent_trades = result['trades'].tail(3)
            for _, trade in recent_trades.iterrows():
                print(f"  {trade['timestamp']} | {trade['symbol']} | {trade['type']} | Size: {trade.get('size', 'N/A')} | PnL: {trade.get('pnl', 'N/A')}")
        
        # Save results with comprehensive QuantStats-Lumi tear sheets
        if args.save and save_dir is not None:
            visualizer.save_results(
                portfolio_returns=portfolio_returns,
                portfolio_metrics=portfolio_metrics,
                asset_results=asset_results,
                trades_df=result["trades"],
                save_dir=save_dir,
                start_date=start_dt,
                end_date=end_dt,
                benchmark_symbol=symbol_names[0] if symbol_names else None
            )
        
        
        # Use final equity from quantstats-lumi metrics
        final_equity_display = portfolio_metrics.get("Final Equity", result["final_cash"])
        
    except ImportError as e:
        print(f"[QuantStats-Lumi] Required packages not installed: {e}")
        print("Please install: pip install quantstats-lumi")
        final_equity_display = result["final_cash"]
        print(f"\nBack-test complete (Basic Mode)")
        print(f"Trades executed: {result['trade_count']}")
        print(f"Final equity: {final_equity_display:.2f} USDT")
    except Exception as e:
        print(f"[QuantStats-Lumi] Could not generate visualizations: {e}")
        final_equity_display = result["final_cash"]
        print(f"\nBack-test complete (Basic Mode)")
        print(f"Trades executed: {result['trade_count']}")
        print(f"Final equity: {final_equity_display:.2f} USDT")

    # Success message for QuantStats-Lumi mode
    if 'portfolio_metrics' in locals() and portfolio_metrics:
        print(f"\n🎉 Back-test complete with QuantStats-Lumi analytics!")
    
    if args.save and save_dir is not None:
        print(f"📁 Results saved to {save_dir}")

    # Display recent trades if available
    if result['trade_count'] > 0 and not ('portfolio_metrics' in locals() and portfolio_metrics):
        print(f"\nLast trades:\n{result['trades'].tail()}")

    # Finished – comprehensive analysis shown above.
