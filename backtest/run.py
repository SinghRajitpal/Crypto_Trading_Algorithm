import asyncio
import argparse
from datetime import datetime, UTC
import os

import config
from backtest.ridge_layer import RidgeLayerSelector
from backtest.backtesting_engine import WalkForwardBacktester
from data.data_engine import DataEngine
from binance_exchange import BinanceClient


async def main():
    parser = argparse.ArgumentParser(description="Run Layer A (ridge selection) and Layer B backtest.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (default: config.DEFAULT_UNIVERSE)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="backtest_output", help="Output directory")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else config.DEFAULT_UNIVERSE
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

    # Layer A: select k per asset
    data_engine = DataEngine(binance_client=BinanceClient(testnet=True), max_candles=2000)
    selector = RidgeLayerSelector()
    ridge_spec = selector.select(data_engine, symbols)

    # Layer B: walk-forward backtest
    backtester = WalkForwardBacktester(symbols, start, end, ridge_spec, initial_capital=10_000.0, output_dir=args.output)
    metrics = await backtester.run()
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "metrics.txt"), "w") as f:
        f.write(str(metrics))


if __name__ == "__main__":
    asyncio.run(main())
