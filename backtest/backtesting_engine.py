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
                    await self.broker.close_position(sym)

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

        # Close any open positions to realise PnL
        await self.broker.close_all_positions()

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
            "trade_count": len(trades),
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
    parser.add_argument("symbols", nargs="?", default="BTCUSDT", help="Comma separated trading pairs, default BTCUSDT")
    parser.add_argument("--tf", default="5m", help="Timeframe, default 5m")
    parser.add_argument("--days", type=int, default=3, help="Days of history, default 3")
    args = parser.parse_args()

    symbols = [(sym.strip().upper(), args.tf) for sym in args.symbols.split(",")]
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

    result = asyncio.run(engine.run())

    # Visualisation / stats -------------------------------------------------------
    try:
        import vectorbt as vbt  # noqa: F401
        from backtest.vectorbt_adapter import load_close_series, portfolio_from_trades

        first_sym = symbols[0][0]

        # Load price data limited to the requested --days window so that
        # vectorbt statistics match the user-specified period.
        close_ser = load_close_series(first_sym, args.tf, start=start_dt, end=end_dt)
        pf = portfolio_from_trades(result["trades"], close_ser, 10_000)
        print(pf.stats())

        # ----------------------------------------------------------
        # Which subplots we want
        wanted = [
            'orders',      # individual order markers
            'trades',      # entry/exit markers
            'value',       # cumulative equity curve (cum-returns)
            'assets',      # asset market value
            'drawdowns'    # under-water curve
        ]

        # Only keep names that exist in this vbt version
        available = set(pf.subplots.keys())
        subplots = [s for s in wanted if s in available]

        # Build figure   (group_by=False avoids the "does not support grouped
        #                 data" warnings you saw for orders/trades panels)
        fig = pf.plots(subplots=subplots, group_by=False)

        fig.show()                       # or fig.write_html("report.html")
        # ----------------------------------------------------------
    except ImportError:
        print("[Vectorbt] vectorbt not installed – install with 'pip install vectorbt' to view plots and stats")
    except Exception as e:
        print(f"[Vectorbt] Could not build or plot portfolio: {e}")

    print("\nBack-test complete")
    print(f"Trades executed : {result['trade_count']}")
    print(f"Final equity    : {result['final_cash']:.2f} USDT")
    if result['trade_count']:
        print("Last trades:\n", result['trades'].tail())

    print(pf.subplots.keys())
