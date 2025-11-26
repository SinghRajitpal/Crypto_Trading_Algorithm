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
from backtest.ridge_layer import RidgeLayerSelector
from backtest.backtesting_engine import WalkForwardBacktester
from backtest.ridge_layer import RidgeLayerResult
from data.data_engine import DataEngine
from data.historical_data import HistoricalDataFetcher
from binance_exchange import BinanceClient


async def main():
    parser = argparse.ArgumentParser(description="Run Layer A (ridge selection) and/or Layer B backtest.")
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
        help="Which layer(s) to run. A: ridge selection only, B: backtest only, both: sequential.",
    )
    parser.add_argument(
        "--ridge-spec-file",
        default=None,
        help="Path to save (Layer A) or load (Layer B) ridge spec JSON. Defaults to <output>/ridge_spec.json",
    )
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.output)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else config.DEFAULT_UNIVERSE
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    ridge_spec_path = args.ridge_spec_file or os.path.join(output_dir, "ridge_spec.json")

    ridge_spec = None

    if args.layer in ("A", "both"):
        selector = RidgeLayerSelector()
        # Keep the seed DataEngine buffer modest; per-TF engines built inside Layer A are sized precisely.
        if hasattr(selector, "_estimate_buffer") and hasattr(selector, "tf_candidates"):
            max_buffer = max(
                selector._estimate_buffer(
                    tf,
                    None,
                    config.W_CANDIDATES_BY_TF.get(
                        tf, [selector.train_min, max(selector.train_min * 2, selector.val_len)]
                    ),
                )
                for tf in selector.tf_candidates
            )
        else:
            max_buffer = max(config.REGRESSION_MIN_TRAIN + config.REGRESSION_VAL_WINDOW + config.REGRESSION_EMBARGO_BARS + 100, 50_000)
        data_engine = DataEngine(binance_client=BinanceClient(testnet=True), max_candles=max_buffer)
        ridge_spec = selector.select(data_engine, symbols, start=start, end=end)
        os.makedirs(os.path.dirname(ridge_spec_path) or ".", exist_ok=True)
        ridge_spec.to_json(ridge_spec_path)
        if args.layer == "A":
            return

    if args.layer in ("B", "both"):
        if ridge_spec is None:
            if not os.path.exists(ridge_spec_path):
                raise FileNotFoundError(f"Ridge spec file not found: {ridge_spec_path}")
            ridge_spec = RidgeLayerResult.from_json(ridge_spec_path)
        backtester = WalkForwardBacktester(
            symbols, start, end, ridge_spec, initial_capital=10_000.0, output_dir=output_dir
        )
        metrics = await backtester.run()
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write(str(metrics))


async def _seed_data_engine(data_engine: DataEngine, symbols, start: datetime, end: datetime) -> None:
    """Load historical OHLCV into the data engine so Layer A can fit ridge."""
    fetcher = HistoricalDataFetcher(testnet=False)
    try:
        lookback_start = _lookback_start(start, config.PRIMARY_TIMEFRAME)
        ohlcv_data = await asyncio.gather(
            *[fetcher.download_ohlcv(sym, config.PRIMARY_TIMEFRAME, lookback_start, end, force=False) for sym in symbols]
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
                    sym, config.PRIMARY_TIMEFRAME, bar
                )
        data_engine.process_all_latest_bars(config.PRIMARY_TIMEFRAME)


def _lookback_start(start: datetime, timeframe: str) -> datetime:
    """Compute training lookback start; prefer timeframe-specific lookback, fallback to global."""
    tf_map = getattr(config, "TIMEFRAME_LOOKBACK_DAYS", {})
    days = tf_map.get(timeframe)
    if days is None:
        days = getattr(config, "RIDGE_TRAIN_LOOKBACK_DAYS", None)
    if days is None:
        return start
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
from datetime import timedelta
