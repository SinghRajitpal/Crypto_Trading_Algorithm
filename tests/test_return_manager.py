import numpy as np

from data.return_manager import ReturnManager


def build_bars():
    bars = []
    ts = 1_000_000
    price = 100.0
    for i in range(60):
        price *= 1.01
        bars.append(
            {
                "timestamp": ts + i * 300_000,
                "open": price / 1.01,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 100 + i,
            }
        )
    return bars


def test_feature_matrix_shape_and_columns():
    rm = ReturnManager(regression_window=60, risk_window=60)
    bars = build_bars()
    for bar in bars:
        rm.update("SYM", bar)
    X, y, ts, cols = rm.get_feature_matrix("SYM", window=50, exclude_outliers=True)
    assert X.shape[0] == y.shape[0] == len(ts)
    assert len(cols) == X.shape[1]
    # Includes momentum/vol/range/turnover/time-of-day
    for name in ["ret_lag1", "vol_lag1", "mom_3", "rv_6", "range_6", "turnover_6", "tod_sin", "tod_cos"]:
        assert name in cols


def test_outlier_exclusion():
    rm = ReturnManager(regression_window=10, risk_window=10)
    bars = build_bars()
    # Inject an outlier bar
    bars[5]["close"] = bars[5]["close"] * 5
    for bar in bars:
        rm.update("SYM", bar)
    X_inc, y_inc, ts_inc, _ = rm.get_feature_matrix("SYM", window=15, exclude_outliers=False)
    X_exc, y_exc, ts_exc, _ = rm.get_feature_matrix("SYM", window=15, exclude_outliers=True)
    assert X_exc.shape[0] <= X_inc.shape[0]
    assert len(ts_exc) <= len(ts_inc)
