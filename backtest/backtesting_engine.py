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
import warnings

import pandas as pd
import numpy as np
import json  # Needed for writing summary.json when --save is enabled
from utils.logging_config import get_logger, console_log

# Only suppress FutureWarning, keep RuntimeWarnings visible to fix underlying issues
warnings.filterwarnings("ignore", category=FutureWarning)

# Get logger for this module
logger = get_logger(__name__)

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
        # Validate date range
        if start >= end:
            raise ValueError(f"Start date {start} must be before end date {end}")
        
        self.symbols = symbols
        # Build helper mapping symbol -> list[timeframe] for quick lookup
        self._symbol_tfs: Dict[str, List[str]] = {}
        for sym, tf in symbols:
            self._symbol_tfs.setdefault(sym, []).append(tf)
        self.strategy = strategy
        self.start = start
        self.end = end
        self.initial_capital = initial_capital  # Store for later use

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
        console_log(f"📥 Loading historical data for {len(self.symbols)} symbol/timeframe pairs...", "INFO")
        
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
            console_log(f"🔍 Processing funding data for {sym}: type={type(series)}, len={len(series) if hasattr(series, '__len__') else 'N/A'}", "DEBUG")
            
            self._funding_data[sym] = series
            if isinstance(series, pd.Series) and not series.empty:
                console_log(f"✅ Loaded {len(series)} funding records for {sym} (from {series.index.min()} to {series.index.max()})", "INFO")
            else:
                console_log(f"⚠️ Empty or invalid funding data for {sym}: {type(series)} - {series}", "WARNING")
            
        console_log(f"✅ Historical data loaded successfully", "INFO")

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
                console_log(f"🔍 Checking funding at {ts} (UTC)", "DEBUG")
                
                # Iterate through all symbols that have funding data
                for sym in self._funding_data.keys():
                    series = self._funding_data.get(sym)
                    if series is None:
                        console_log(f"⚠️ No funding data for {sym}", "DEBUG")
                        continue
                    
                    # Check if this exact timestamp exists in funding data
                    if ts not in series.index:
                        console_log(f"⚠️ Timestamp {ts} not found in funding data for {sym}", "DEBUG")
                        continue
                    
                    rate = series.loc[ts]
                    console_log(f"💰 Applying funding for {sym}: rate={rate:.6f}", "DEBUG")
                    
                    # Apply funding and log the result
                    payment = await self.broker.apply_funding(sym, rate)
                    if payment != 0:
                        console_log(f"✅ Funding applied: {sym} paid ${payment:.4f}", "INFO")

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
            "total_return_pct": float((final_equity / self.initial_capital - 1) * 100),
            "max_drawdown_pct": 0.0,  # Placeholder - would need trade history to calculate properly
        }

    @staticmethod
    def _print_cost_analysis(trades_df: pd.DataFrame):
        """Print detailed cost breakdown analysis."""
        if trades_df.empty:
            return
            
        console_log("\n" + "="*80, "INFO")
        console_log("TRADING COST ANALYSIS", "INFO")
        console_log("="*80, "INFO")
        
        # Separate trade types
        open_trades = trades_df[trades_df['type'] == 'open']
        close_trades = trades_df[trades_df['type'] == 'close']
        funding_trades = trades_df[trades_df['type'] == 'funding']
        
        # Calculate total costs
        total_trading_fees = (open_trades['fee'].sum() + close_trades['fee'].sum()) if 'fee' in trades_df.columns else 0
        total_funding_costs = abs(funding_trades['payment'].sum()) if 'payment' in funding_trades.columns and not funding_trades.empty else 0
        total_notional_volume = (open_trades['notional'].sum() + close_trades['notional'].sum()) if 'notional' in trades_df.columns else 0
        
        # Cost as percentage of volume
        trading_fee_pct = (total_trading_fees / total_notional_volume * 100) if total_notional_volume > 0 else 0
        funding_cost_pct = (total_funding_costs / total_notional_volume * 100) if total_notional_volume > 0 else 0
        total_cost_pct = trading_fee_pct + funding_cost_pct
        
        console_log(f"Total Trading Volume (USDT)       ${total_notional_volume:>12,.2f}", "INFO")
        console_log(f"Total Trading Fees (USDT)         ${total_trading_fees:>12,.2f}", "INFO")
        console_log(f"Total Funding Costs (USDT)        ${total_funding_costs:>12,.2f}", "INFO")
        console_log(f"Total Transaction Costs (USDT)    ${total_trading_fees + total_funding_costs:>12,.2f}", "INFO")
        
        console_log("\n" + "-"*80, "INFO")
        console_log("COST BREAKDOWN BY PERCENTAGE", "INFO")
        console_log("-"*80, "INFO")
        
        console_log(f"Trading Fees (% of volume)         {trading_fee_pct:>12.3f}%", "INFO")
        console_log(f"Funding Costs (% of volume)        {funding_cost_pct:>12.3f}%", "INFO")
        console_log(f"Total Costs (% of volume)          {total_cost_pct:>12.3f}%", "INFO")
        console_log(f"", "INFO")
        console_log(f"Note: Trading fees include spreads and slippage costs", "INFO")
        
        # Trade-level analysis
        if not open_trades.empty:
            avg_trade_size = open_trades['notional'].mean()
            avg_trade_fee = open_trades['fee'].mean()
            
            console_log("\n" + "-"*80, "INFO")
            console_log("PER-TRADE COST ANALYSIS", "INFO")
            console_log("-"*80, "INFO")
            
            console_log(f"Number of Trades                   {len(open_trades):>12,}", "INFO")
            console_log(f"Average Trade Size (USDT)          ${avg_trade_size:>12,.2f}", "INFO")
            console_log(f"Average Trading Fee per Trade      ${avg_trade_fee:>12,.2f}", "INFO")
            console_log(f"Fee as % of Avg Trade Size         {(avg_trade_fee/avg_trade_size*100):>12.3f}%", "INFO")
        
        # Funding analysis
        if not funding_trades.empty:
            console_log("\n" + "-"*80, "INFO")
            console_log("FUNDING COST ANALYSIS", "INFO")
            console_log("-"*80, "INFO")
            
            funding_count = len(funding_trades)
            avg_funding_payment = abs(funding_trades['payment'].mean())
            total_funding_hours = funding_count * 8  # 8 hours per funding period
            
            console_log(f"Number of Funding Periods          {funding_count:>12,}", "INFO")
            console_log(f"Total Hours with Open Positions    {total_funding_hours:>12,}", "INFO")
            console_log(f"Average Funding Rate               {funding_trades['rate'].mean()*100:>12.3f}%", "INFO")
            console_log(f"Average Funding Payment (USDT)     ${avg_funding_payment:>12,.2f}", "INFO")
        
        console_log("="*80, "INFO")

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
    parser.add_argument("--individual-plots", action="store_true", help="Generate individual asset HTML reports (default: False)")
    args = parser.parse_args()

    if not args.symbols.strip() or args.symbols.strip().upper() == "ALL":
        import config
        symbols = [(sym.upper(), args.tf) for sym, _ in config.symbols]
        logger.info(f"Running ALL configured symbols: {', '.join([s for s, _ in symbols])}")
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

    logger.info(f"Strategy: {strategy.strategy_id}")
    
    # Console log: Backtest initialization
    console_log(f"🚀 Starting backtest for {len(symbols)} symbols using {strategy.strategy_id} strategy", "INFO")
    console_log(f"📅 Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}", "INFO")
    if args.individual_plots:
        console_log("📊 Individual asset HTML reports: ENABLED", "INFO")
    else:
        console_log("📊 Individual asset HTML reports: DISABLED (use --individual-plots to enable)", "INFO")

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
    console_log("📊 Loading historical data...", "INFO")
    result = asyncio.run(engine.run())
    console_log("✅ Backtest execution completed", "INFO")

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

        logger.info(f"Saving report to {save_dir}")

    # Visualization / stats with quantstats-lumi only -------------------------
    try:
        from backtest.visualizer import QuantStatsVisualizer
        
        console_log("📈 Calculating portfolio metrics...", "INFO")
        logger.info("\nPORTFOLIO STATISTICS (QuantStats-Lumi):")
        
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
        
        console_log("📊 Analyzing individual assets...", "INFO")
        # Generate individual asset analysis (only calculate metrics, not HTML reports)
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
                logger.warning(f"Could not analyze asset {symbol}: {e}")
        
        console_log("📋 Generating performance summary...", "INFO")
        
        # Calculate and display cost analysis
        BacktestingEngine._print_cost_analysis(result["trades"])
        
        # Print comprehensive metrics summary using QuantStats-Lumi
        visualizer.print_summary(portfolio_metrics, {k: v[1] for k, v in asset_results.items()})
        
        # Additional trade information
        final_equity_display = portfolio_metrics.get("Final Equity", result["final_cash"])
        logger.info(f"\n📊 TRADE SUMMARY")
        logger.info(f"Trades executed: {result['trade_count']}")
        logger.info(f"Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        if result['trade_count'] > 0:
            logger.info(f"\n📋 RECENT TRADES:")
            recent_trades = result['trades'].tail(3)
            for _, trade in recent_trades.iterrows():
                logger.info(f"  {trade['timestamp']} | {trade['symbol']} | {trade['type']} | Size: {trade.get('size', 'N/A')} | PnL: {trade.get('pnl', 'N/A')}")
        
        # Save results with comprehensive QuantStats-Lumi tear sheets
        if args.save and save_dir is not None:
            console_log(f"💾 Generating and saving performance reports...", "INFO")
            visualizer.save_results(
                portfolio_returns=portfolio_returns,
                portfolio_metrics=portfolio_metrics,
                asset_results=asset_results,
                trades_df=result["trades"],
                save_dir=save_dir,
                start_date=start_dt,
                end_date=end_dt,
                benchmark_symbol=symbol_names[0] if symbol_names else None,
                generate_individual_plots=args.individual_plots  # Pass the flag
            )
            console_log(f"📁 Reports saved to: {save_dir}", "INFO")
        
        
        # Use final equity from quantstats-lumi metrics
        final_equity_display = portfolio_metrics.get("Final Equity", result["final_cash"])
        
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.info("Please install: pip install quantstats-lumi")
        final_equity_display = result["final_cash"]
        logger.info(f"\nBack-test complete (Basic Mode)")
        logger.info(f"Trades executed: {result['trade_count']}")
        logger.info(f"Final equity: {final_equity_display:.2f} USDT")
    except Exception as e:
        logger.error(f"Could not generate visualizations: {e}")
        final_equity_display = result["final_cash"]
        logger.info(f"\nBack-test complete (Basic Mode)")
        logger.info(f"Trades executed: {result['trade_count']}")
        logger.info(f"Final equity: {final_equity_display:.2f} USDT")

    # Success message for QuantStats-Lumi mode
    if 'portfolio_metrics' in locals() and portfolio_metrics:
        console_log(f"🎉 Backtest completed successfully with advanced analytics!", "INFO")
        final_equity_display = portfolio_metrics.get("Final Equity", result["final_cash"])
        console_log(f"💰 Final Portfolio Value: ${final_equity_display:.2f} USDT", "INFO")
        console_log(f"📊 Total Trades Executed: {result['trade_count']}", "INFO")
        logger.info(f"\n🎉 Back-test complete with QuantStats-Lumi analytics!")
    else:
        console_log(f"🎉 Backtest completed!", "INFO")
        console_log(f"💰 Final Portfolio Value: ${result['final_cash']:.2f} USDT", "INFO")
        console_log(f"📊 Total Trades Executed: {result['trade_count']}", "INFO")
    
    if args.save and save_dir is not None:
        logger.info(f"📁 Results saved to {save_dir}")

    # Display recent trades if available
    if result['trade_count'] > 0 and not ('portfolio_metrics' in locals() and portfolio_metrics):
        logger.info(f"\nLast trades:\n{result['trades'].tail()}")

    # Finished – comprehensive analysis shown above.
