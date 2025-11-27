import asyncio
import argparse
from datetime import datetime, UTC, timedelta
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct execution without PYTHONPATH.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from backtest.gru_layer import GRULayerTrainer, GRULayerResult
from data.data_engine import DataEngine
from data.historical_data import HistoricalDataFetcher
from binance_exchange import BinanceClient


async def main():
    parser = argparse.ArgumentParser(description="Run Layer A GRU training and/or Layer B GRU backtest.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (default: config.DEFAULT_UNIVERSE)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=None,
        help="Run output directory. If omitted, auto-creates backtest/backtest_results/<date>_trialNN.",
    )
    parser.add_argument(
        "--layer",
        choices=["A", "B", "both"],
        default="both",
        help="Which layer(s) to run. A: GRU training only, B: backtest only, both: sequential.",
    )
    parser.add_argument(
        "--gru-model-dir",
        default=None,
        help="Path to a trained GRU model directory (uses Layer A output if not provided).",
    )
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.output)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else config.DEFAULT_UNIVERSE
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    gru_spec_path = os.path.join(output_dir, "gru_spec.json")
    gru_model_dir = args.gru_model_dir

    gru_spec: GRULayerResult | None = None

    if args.layer in ("A", "both"):
        artifacts_dir = os.path.join(output_dir, "layerA_artifacts")
        trainer = GRULayerTrainer(symbols[0], timeframe=config.GRU_TIMEFRAME)
        gru_spec = await trainer.train(start=start, end=end, output_dir=artifacts_dir)
        gru_model_dir = gru_spec.model_dir
        os.makedirs(os.path.dirname(gru_spec_path) or ".", exist_ok=True)
        gru_spec.to_json(gru_spec_path)
        if args.layer == "A":
            return

    if args.layer in ("B", "both"):
        # Import backtester lazily to avoid heavy imports (matplotlib) when only running Layer A.
        from backtest.backtesting_engine import WalkForwardBacktester
        if gru_model_dir is None:
            if os.path.exists(gru_spec_path):
                gru_spec = GRULayerResult.from_json(gru_spec_path)
                gru_model_dir = gru_spec.model_dir
        if gru_model_dir is None:
            raise FileNotFoundError("GRU model directory not provided and gru_spec.json not found.")
        backtester = WalkForwardBacktester(
            symbols, start, end, gru_model_dir=gru_model_dir, initial_capital=10_000.0, output_dir=output_dir
        )
        metrics = await backtester.run()
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write(str(metrics))


async def _seed_data_engine(data_engine: DataEngine, symbols, start: datetime, end: datetime) -> None:
    """Load historical OHLCV into the data engine so Layer A can fit GRU."""
    fetcher = HistoricalDataFetcher(testnet=False)
    try:
        lookback_start = _lookback_start(start, config.GRU_TIMEFRAME)
        ohlcv_data = await asyncio.gather(
            *[fetcher.download_ohlcv(sym, config.GRU_TIMEFRAME, lookback_start, end, force=False) for sym in symbols]
        )
    finally:
        # Ensure the aiohttp/binance client is closed to avoid unclosed-session warnings.
        await fetcher.close()

    data_map = {sym: df for sym, df in zip(symbols, ohlcv_data) if df is not None}
    clock = sorted({ts for df in data_map.values() for ts in df.index})
    for ts in clock:
        for sym, df in data_map.items():
            if ts in df.index:
                candle = df.loc[ts]
                bar = [int(ts.timestamp() * 1000)] + candle.tolist()
                await data_engine.data_fetcher.data_processor.update_tracked_candles(
                    sym, config.GRU_TIMEFRAME, bar
                )
        data_engine.process_all_latest_bars(config.GRU_TIMEFRAME)


def _lookback_start(start: datetime, timeframe: str) -> datetime:
    """Compute training lookback start for GRU window with cushion."""
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


def _resolve_output_dir(output_arg: str | None) -> str:
    """Determine where to store run artifacts; default under backtest/backtest_results/<date>_trialNN."""
    if output_arg:
        return os.path.abspath(output_arg)

    base_dir = os.path.join(os.path.dirname(__file__), "backtest_results")
    os.makedirs(base_dir, exist_ok=True)
    date_prefix = datetime.now(UTC).strftime("%Y-%m-%d")
    existing = [d for d in os.listdir(base_dir) if d.startswith(f"{date_prefix}_trial")]
    trial_nums = []
    for name in existing:
        try:
            trial_part = name.split("_trial")[-1]
            trial_nums.append(int(trial_part))
        except ValueError:
            continue
    next_trial = (max(trial_nums) + 1) if trial_nums else 1
    run_name = f"{date_prefix}_trial{next_trial:02d}"
    return os.path.join(base_dir, run_name)


if __name__ == "__main__":
    asyncio.run(main())
