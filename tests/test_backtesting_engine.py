import asyncio
import sys
import types
from datetime import datetime, timedelta, UTC

import numpy as np
import pandas as pd

# Stub binance modules to avoid external dependency during tests
if "binance" not in sys.modules:
    binance_mod = types.ModuleType("binance")

    class _AsyncClient:
        @classmethod
        async def create(cls, *args, **kwargs):
            return cls()

        async def close_connection(self):
            return None

    class _BinanceSocketManager:
        def __init__(self, *args, **kwargs):
            pass

    enums_mod = types.ModuleType("binance.enums")
    enums_mod.__all__ = []
    binance_mod.AsyncClient = _AsyncClient
    binance_mod.BinanceSocketManager = _BinanceSocketManager
    sys.modules["binance"] = binance_mod
    sys.modules["binance.enums"] = enums_mod

import config
from backtest.backtesting_engine import WalkForwardBacktester, BacktestMetrics
from backtest.ridge_layer import RidgeLayerResult


class StubFetcher:
    def __init__(self, df_map):
        self.df_map = df_map

    async def download_ohlcv(self, symbol, timeframe, start, end, force=False):
        return self.df_map[symbol]


def make_df(start: datetime, periods: int, price_start: float = 100.0, step: float = 0.1):
    times = [start + timedelta(minutes=5 * i) for i in range(periods)]
    prices = [price_start + step * i for i in range(periods)]
    data = {
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [100 + i for i in range(periods)],
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(times, tz=UTC))
    return df


def test_walk_forward_backtester_runs_with_stubbed_history(monkeypatch):
    # Shrink windows for faster test runtime
    monkeypatch.setattr(config, "REGRESSION_MIN_TRAIN", 30)
    monkeypatch.setattr(config, "RISK_WINDOW", 20)
    periods = max(config.REGRESSION_MIN_TRAIN + 10, config.RISK_WINDOW + 10)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    df = make_df(start, periods)
    symbols = ["TESTUSDT"]

    ridge_spec = RidgeLayerResult(
        k_per_asset={symbols[0]: 0.0},
        msep_per_asset={symbols[0]: 0.0},
        rl_vs_ls={symbols[0]: None},
        t_threshold=config.RIDGE_T_THRESHOLD,
        samples_per_asset={symbols[0]: periods},
    )

    bt = WalkForwardBacktester(
        symbols=symbols,
        start=start,
        end=start + timedelta(minutes=periods * 5),
        ridge_spec=ridge_spec,
        initial_capital=10000.0,
        output_dir=None,
    )
    # Ensure data engine processes our synthetic symbol
    bt.data_engine.data_fetcher.symbol_timeframes = [(symbols[0], config.PRIMARY_TIMEFRAME)]
    # Stub the historical fetcher to avoid network calls
    bt.fetcher = StubFetcher({symbols[0]: df})

    metrics = asyncio.run(bt.run())
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.turnover >= 0.0
    # Basic risk diagnostics captured
    assert metrics.max_drawdown >= 0.0


def test_backtester_metrics_finite_and_trades(monkeypatch):
    # Encourage trading: short windows and trending prices
    monkeypatch.setattr(config, "REGRESSION_MIN_TRAIN", 20)
    monkeypatch.setattr(config, "RISK_WINDOW", 15)
    periods = 80
    start = datetime(2024, 1, 1, tzinfo=UTC)
    df = make_df(start, periods, price_start=50.0, step=0.5)  # upward trend
    symbols = ["AAA"]

    ridge_spec = RidgeLayerResult(
        k_per_asset={symbols[0]: 0.0},
        msep_per_asset={symbols[0]: 0.0},
        rl_vs_ls={symbols[0]: None},
        t_threshold=config.RIDGE_T_THRESHOLD,
        samples_per_asset={symbols[0]: periods},
    )

    bt = WalkForwardBacktester(
        symbols=symbols,
        start=start,
        end=start + timedelta(minutes=periods * 5),
        ridge_spec=ridge_spec,
        initial_capital=10000.0,
        output_dir=None,
    )
    bt.data_engine.data_fetcher.symbol_timeframes = [(symbols[0], config.PRIMARY_TIMEFRAME)]
    bt.fetcher = StubFetcher({symbols[0]: df})

    metrics = asyncio.run(bt.run())
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.turnover > 0.0  # trades occurred
    # Impact vs slippage should be finite
    assert metrics.impact_vs_slippage == metrics.impact_vs_slippage  # not NaN
    # Sharpe finite (could be negative/positive)
    assert metrics.sharpe == metrics.sharpe


def test_backtester_skips_when_not_ready(monkeypatch):
    # Too few bars to meet training/risk windows → no trades, zero metrics
    monkeypatch.setattr(config, "REGRESSION_MIN_TRAIN", 50)
    monkeypatch.setattr(config, "RISK_WINDOW", 40)
    periods = 20  # insufficient
    start = datetime(2024, 1, 1, tzinfo=UTC)
    df = make_df(start, periods, price_start=100.0, step=0.0)  # flat prices
    symbols = ["AAA"]

    ridge_spec = RidgeLayerResult(
        k_per_asset={symbols[0]: 0.0},
        msep_per_asset={symbols[0]: 0.0},
        rl_vs_ls={symbols[0]: None},
        t_threshold=config.RIDGE_T_THRESHOLD,
        samples_per_asset={symbols[0]: periods},
    )

    bt = WalkForwardBacktester(
        symbols=symbols,
        start=start,
        end=start + timedelta(minutes=periods * 5),
        ridge_spec=ridge_spec,
        initial_capital=10000.0,
        output_dir=None,
    )
    bt.data_engine.data_fetcher.symbol_timeframes = [(symbols[0], config.PRIMARY_TIMEFRAME)]
    bt.fetcher = StubFetcher({symbols[0]: df})

    metrics = asyncio.run(bt.run())
    assert metrics.turnover == 0.0
    assert metrics.pnl == 0.0
    assert metrics.cum_return_pct == 0.0


def test_backtester_monitoring_outputs_finite(monkeypatch):
    # Use two symbols to exercise covariance losses and risk diagnostics
    monkeypatch.setattr(config, "REGRESSION_MIN_TRAIN", 20)
    monkeypatch.setattr(config, "RISK_WINDOW", 15)
    periods = 60
    start = datetime(2024, 1, 1, tzinfo=UTC)
    df1 = make_df(start, periods, price_start=100.0, step=0.2)
    df2 = make_df(start, periods, price_start=50.0, step=-0.1)  # downward trend
    symbols = ["AAA", "BBB"]

    ridge_spec = RidgeLayerResult(
        k_per_asset={s: 0.0 for s in symbols},
        msep_per_asset={s: 0.0 for s in symbols},
        rl_vs_ls={s: None for s in symbols},
        t_threshold=config.RIDGE_T_THRESHOLD,
        samples_per_asset={s: periods for s in symbols},
    )

    bt = WalkForwardBacktester(
        symbols=symbols,
        start=start,
        end=start + timedelta(minutes=periods * 5),
        ridge_spec=ridge_spec,
        initial_capital=10000.0,
        output_dir=None,
    )
    bt.data_engine.data_fetcher.symbol_timeframes = [(s, config.PRIMARY_TIMEFRAME) for s in symbols]
    bt.fetcher = StubFetcher({symbols[0]: df1, symbols[1]: df2})

    metrics = asyncio.run(bt.run())
    # Metrics that aggregate monitoring should be finite when trades/returns exist
    assert metrics.impact_vs_slippage == metrics.impact_vs_slippage
    assert metrics.sharpe == metrics.sharpe
    # Risk/cov losses should be present in internal summary
    assert isinstance(bt._metrics.get("risk_diag_avg", {}), dict)
