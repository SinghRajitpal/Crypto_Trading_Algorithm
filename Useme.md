# Usage Guide

This is a guide which provided step by step instructions on how to use this code. Also, when running this code for the first time, it takes a lot of time to import the libraries and the data but after the first run, it runs quicker.

## 1) One-Time Setup
1. Install Python 3.11+.
2. From the repo root, create a virtual environment:  
   `python -m venv .venv`
3. Activate it:  
   - macOS/Linux: `source .venv/bin/activate`  
   - Windows: `.\\.venv\\Scripts\\activate`
4. Install dependencies:  
   `pip install -r requirements.txt`


## 2) Configure Binance Demo Keys
1. In Binance, switch Futures to Demo/Testnet and create demo API keys.
2. Open `config.py` and set:
   - `binance_futures_demo["demo_api_key"] = "<your_demo_key>"`
   - `binance_futures_demo["demo_api_secret"] = "<your_demo_secret>"`
3. Keep keys out of version control.


## 3) Know the Key Files
- `config.py`: credentials, universe/timeframe, optimizer/risk/Kelly settings, manifest/GRU paths.
- `main.py`: live/demo trading entrypoint.
- `backtest/backtest_results/`: outputs from Layer A/B backtests (predictions, manifest, metrics, plots).


## 4) Live / Demo Trading (simple)
1. Ensure `config.py` has your demo keys.
2. If you have a another model to use, set `LAYERA_MANIFEST_PATH` to its path. Otherwise the system falls back to `GRU_MODEL_DIR` when `FORECAST_FALLBACK` allows.
3. Check universe/timeframe in `config.py`: `DEFAULT_UNIVERSE`, `PRIMARY_TIMEFRAME` (and `GRU_TIMEFRAME`).
4. Run demo trading from the repo root:  
   `python main.py`
5. Stop with Ctrl+C.
6. Outputs: console logs, `logs/monitoring.jsonl`, cached data under `data/cache`.
7. For real trading (not demo): instantiate `TradingAlgorithm(demo=False)` in `main.py` and fill `binance_futures` with real keys. Double-check risk/exposure settings before doing this (not recommended as this is not a stable version for live trading).


## 5) Backtesting Workflows

### A) Layer A — GRU Training & Predictions
What it does: trains walk-forward GRU per symbol and writes predictions/metrics/plots plus a manifest you can use live.

Run (example):
```
python -m backtest.run --start 2023-01-01 --end 2023-06-01 --layer A \
  --symbols BTCUSDT,ETHUSDT --output backtest/backtest_results/demo_run
```
Key knobs in `config.py`: `GRU_TIMEFRAME`, `GRU_LOOKBACK`, `GRU_FEATURE_SCHEMA`, `GRU_RETRAIN_DAYS`, `GRU_MIN_TRAIN_SAMPLES`, `GRU_DEVICE`.

Outputs (under `output/layerA_artifacts/`):
- Per-symbol dirs with `predictions.csv`, `metrics.jsonl`, `summary.json`, checkpoints, plots.
- `layerA_manifest.json` (point `config.LAYERA_MANIFEST_PATH` here for live/demo).

### B) Layer B — Portfolio Backtest (uses Layer A predictions)
What it does: loads predictions and simulates the live portfolio stack with risk/optimizer/Kelly.

Run with explicit predictions:
```
python -m backtest.run --start 2023-01-01 --end 2023-06-01 --layer B \
  --output backtest/backtest_results/demo_run \
  --predictions path/to/BTCUSDT/predictions.csv,path/to/ETHUSDT/predictions.csv
```
If `--predictions` is omitted, the script auto-discovers `predictions.csv` under `--output/layerA_artifacts/` if present.

Key knobs: risk/optimizer/Kelly in `config.py` (`RISK_WINDOW`, `EWMA_LAMBDA`, `RISK_AVERSION`, `WEIGHT_MIN/WEIGHT_MAX`, `MAX_NET_EXPOSURE`, `MAX_GROSS_EXPOSURE`, `TURNOVER_PENALTY_LAMBDA`, `FEE_RATE`, `SLIPPAGE_BPS_DEFAULT`, `KELLY_*`). Optional benchmark via `BENCHMARK_SYMBOL`.

Outputs (under `output/layerB_artifacts/`):
- `summary.json`, `metrics.csv`, `equity.png`, `risk_diag.png`, `kelly.png` (if applicable), `risk_series.jsonl`, `equity.csv`.
- Benchmark artifacts (if enabled) under `benchmark_artifacts/`.

### C) Both Layers Sequentially
Run both in one shot:
```
python -m backtest.run --start 2023-01-01 --end 2023-06-01 \
  --layer both --symbols BTCUSDT,ETHUSDT --output backtest/backtest_results/demo_run
```
Layer A artifacts feed Layer B automatically; a manifest is emitted for live/demo use.


## 6) Quick CLI Reference
- `--symbols`: comma-separated list (default: `config.DEFAULT_UNIVERSE`)
- `--start`, `--end`: `YYYY-MM-DD` (required)
- `--output`: output dir (default auto-creates `backtest/backtest_results/<date>_trialNN`)
- `--layer`: `A`, `B`, or `both` (default `both`)
- `--predictions`: comma-separated `predictions.csv` paths (needed if running `--layer B` without Layer A outputs under `--output`)


## 7) Troubleshooting
- No live forecasts: ensure `LAYERA_MANIFEST_PATH` exists or `GRU_MODEL_DIR` has checkpoints; check logs for missing bars/features.
- Auth errors: verify demo keys and that you are on demo/testnet.
- Backtest missing predictions: pass `--predictions` or point `--output` to a dir containing `layerA_artifacts`.
- Headless plot issues: metrics/CSVs still write even if plot rendering fails.


## 8) End-to-End Example (Layer A → Layer B → Demo) (recommended for you Mr.Ignazio):

Use one output directory to keep artifacts together; then point live/demo at the produced manifest.

**Step 1: Backtest Layer A (train GRUs, produce manifest)**
- Config notes: ensure `GRU_TIMEFRAME`, `GRU_LOOKBACK`, `GRU_FEATURE_SCHEMA` match the data you want; keep demo keys in `config.py` (not used here).
- Command:
  ```
  python -m backtest.run --start 2023-01-01 --end 2023-06-01 --layer A \
    --symbols BTCUSDT,ETHUSDT --output backtest/backtest_results/e2e_run
  ```
- Result: `backtest/backtest_results/e2e_run/layerA_artifacts/layerA_manifest.json` plus per-symbol `predictions.csv`.

**Step 2: Backtest Layer B (portfolio sim using Layer A predictions)**
- Config notes: risk/optimizer/Kelly settings in `config.py` apply here.
- Command (auto-discovers predictions from Step 1):
  ```
  python -m backtest.run --start 2023-01-01 --end 2023-06-01 --layer B \
    --output backtest/backtest_results/e2e_run
  ```
  If you prefer to pass predictions explicitly:
  ```
  python -m backtest.run --start 2023-01-01 --end 2023-06-01 --layer B \
    --output backtest/backtest_results/e2e_run \
    --predictions backtest/backtest_results/e2e_run/layerA_artifacts/BTCUSDT/predictions.csv,\backtest/backtest_results/e2e_run/layerA_artifacts/ETHUSDT/predictions.csv
  ```
- Result: `backtest/backtest_results/e2e_run/layerB_artifacts/summary.json` (plus metrics/plots).

**Step 3: Run Demo Mode using the new manifest**
- Config changes in `config.py`:
  - Set `LAYERA_MANIFEST_PATH = "backtest/backtest_results/e2e_run/layerA_artifacts/layerA_manifest.json"`
  - Ensure `binance_futures_demo` keys are set.
  - Align `DEFAULT_UNIVERSE`/`PRIMARY_TIMEFRAME` with the manifest’s symbols/timeframe.
- Command (from repo root):
  ```
  python main.py
  ```
- Outputs: live/demo loop runs with the trained GRUs; logs to console and `logs/monitoring.jsonl`; caches in `data/cache`.
