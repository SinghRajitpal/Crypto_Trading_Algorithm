# High-Performance Crypto Futures Trading Algorithm

This project covers a trading algorithm designed for crypto perpetual futures on Binance. The aim of this trading setup is to hold the optimal portfolio and therefore rebalance it. For this the mean-variance optimization is used (by Markowitz). One input for the portfolio optimization is the expected returns of an asset. Hence, for forecasting returns the gated recurrent unit (GRU) was used.
Summarised, there are two distinct workflows of this trading setup:
1. Live/Demo trading on Binance Futures
2. Backtest (Layer A, Layer B or both Layers)

Live trading is currently not recommended (as there are components to be updated for this process) but demo trading can be run in order to see algo working. Regarding the backtests, Layer A is responsible for training, making predictions (during backtest window) and forecasing performance and Layer B is responsible for running the entire portfolio logic against historical data.

## Key Features
- Live/demo trading loop with Binance futures client (live is not recommended but demo can be run)
- GRU-based forecast layer with manifest loader (`config.LAYERA_MANIFEST_PATH`) and fallback to legacy GRU checkpoints (`config.GRU_MODEL_DIR`).
- Risk model (EWMA) feeding a mean–variance optimizer with turnover penalty, Kelly overlay, and exposure caps.
- Dynamic universe selection via market-cap ranks and volume filters (should be used for the live worklfow and is not supported for the demo and backtest)
- Walk-forward backtesting: Layer A GRU training/inference, Layer B portfolio simulation with benchmark comparison, plots, and JSON/CSV outputs.
- Historical data saved to `data/cache` for faster downloads as rate throttling from binance api takes significant amount of time.

## Project Structure
- `main.py`: entrypoint for live/demo trading and main loop.
- `config.py`: credentials, universe/timeframe defaults, optimizer/kelly/risk configs, GRU paths
- `algorithm/`: forecast models and strategies.
- `data/`: streaming/polling fetcher, feature engineering, return manager, universe selection, historical downloader/cache.
- `execution/`: risk model, mean–variance optimizer, Kelly overlay, trade generator, order executor, alerts.
- `backtest/`: Layer A trainer, Layer B walk-forward backtester, visualization/reporting helpers.
- `tests/`: pytest coverage for data, execution, and backtesting components.

## Technologies
Python 3.11+, python-binance, PyTorch, NumPy, Pandas, Matplotlib, PyArrow, asyncio, pytest.

## Quickstart
1. `python -m venv .venv`
2. Activate: `source .venv/bin/activate` (Unix/Mac) or `.\\.venv\\Scripts\\activate` (Windows).
3. `pip install -r requirements.txt`
4. Set Binance demo API keys in `config.py` (`binance_futures_demo`)
5. Start demo trading: `python main.py` (uses Binance futures demo endpoint; Ctrl+C to stop).
6. For more details, backtests, layer-specific runs, and parser details, see [USEME.md](USEME.md).
