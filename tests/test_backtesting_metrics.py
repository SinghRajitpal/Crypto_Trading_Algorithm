import numpy as np

from backtest.backtesting_engine import WalkForwardBacktester


def test_bars_per_day():
    assert WalkForwardBacktester._bars_per_day("8h") == 3.0
    assert WalkForwardBacktester._bars_per_day("1d") == 1.0


def test_sharpe_and_volatility():
    rets = [0.01, -0.02, 0.03]
    bars_per_day = 3.0
    sharpe = WalkForwardBacktester._sharpe(rets, bars_per_day)
    vol = WalkForwardBacktester._volatility(rets, bars_per_day)
    # Manual calc
    arr = np.array(rets)
    mean = arr.mean()
    std = arr.std(ddof=1)
    ann = np.sqrt(bars_per_day * 365)
    expected_sharpe = mean / std * ann
    expected_vol = std * ann
    assert np.isclose(sharpe, expected_sharpe)
    assert np.isclose(vol, expected_vol)


def test_max_and_avg_drawdown():
    equity = [100, 110, 105, 120, 115]
    max_dd = WalkForwardBacktester._max_drawdown(equity)
    # Max drawdown occurs from 110 to 105 -> 4.545%
    assert np.isclose(max_dd, (110 - 105) / 110)


def test_avg_win_loss():
    rets = [0.02, -0.01, 0.0, -0.03, 0.01]
    avg_win, avg_loss = WalkForwardBacktester._avg_win_loss(rets)
    assert np.isclose(avg_win, np.mean([0.02, 0.01]))
    assert np.isclose(avg_loss, np.mean([0.01, 0.03]))
