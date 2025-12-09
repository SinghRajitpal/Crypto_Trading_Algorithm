import asyncio
import argparse
import json
from datetime import datetime, UTC, timedelta
import os
import sys
from pathlib import Path
from dataclasses import asdict

# Ensure repository root is on sys.path for direct execution without PYTHONPATH.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from backtest.gru_layer import GRULayerTrainer, GRULayerResult
from data.data_engine import DataEngine


async def main():
    """Parse CLI args and run Layer A training, Layer B backtest, or both."""
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
        "--predictions",
        nargs="?",
        default=None,
        const="",
        help=(
            "Optional comma-separated paths to Layer A predictions.csv files (one per symbol). "
            "If omitted or empty and layer=A|both, uses freshly produced Layer A outputs. "
            "If layer=B only and not provided, tries to auto-discover predictions under the output dir."
        ),
    )
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.output)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else config.DEFAULT_UNIVERSE
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    predictions_paths = _parse_predictions_arg(args.predictions)

    if args.layer in ("A", "both"):
        artifacts_root = os.path.join(output_dir, "layerA_artifacts")
        os.makedirs(artifacts_root, exist_ok=True)
        layer_a_results: list[GRULayerResult] = []
        for sym in symbols:
            sym_dir = os.path.join(artifacts_root, sym)
            trainer = GRULayerTrainer(sym, timeframe=config.GRU_TIMEFRAME)
            result = await trainer.run(start=start, end=end, output_dir=sym_dir)
            layer_a_results.append(result)
            predictions_paths.append(result.predictions_path)
        manifest_path = os.path.join(artifacts_root, "layerA_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump([asdict(r) for r in layer_a_results], f, indent=2)
        if args.layer == "A":
            return

    if args.layer in ("B", "both"):
        # Import backtester lazily to avoid heavy imports (matplotlib) when only running Layer A.
        from backtest.backtesting_engine import WalkForwardBacktester
        if not predictions_paths:
            predictions_paths = _discover_predictions(output_dir)
        if not predictions_paths:
            raise FileNotFoundError(
                "Predictions paths not provided and could not be auto-discovered. "
                "Pass --predictions with comma-separated paths to predictions.csv files "
                "or point --output to a prior Layer A artifacts directory."
            )
        layer_b_dir = os.path.join(output_dir, "layerB_artifacts")
        benchmark_dir = os.path.join(output_dir, "benchmark_artifacts")
        backtester = WalkForwardBacktester(
            symbols,
            start,
            end,
            predictions_paths=predictions_paths,
            initial_capital=10_000.0,
            output_dir=layer_b_dir,
            benchmark_dir=benchmark_dir,
        )
        metrics = await backtester.run()
        os.makedirs(layer_b_dir, exist_ok=True)
        with open(os.path.join(layer_b_dir, "metrics.txt"), "w") as f:
            f.write(str(metrics))


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


def _parse_predictions_arg(raw: str | None) -> list[str]:
    """Parse comma-separated predictions argument into a clean list of paths."""
    if raw is None:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _discover_predictions(output_dir: str) -> list[str]:
    """Find predictions.csv files under the Layer A artifacts directory."""
    artifacts_root = Path(output_dir) / "layerA_artifacts"
    if not artifacts_root.exists():
        return []
    candidates = sorted(artifacts_root.glob("*/predictions.csv"))
    return [str(p) for p in candidates if p.is_file()]


if __name__ == "__main__":
    asyncio.run(main())
