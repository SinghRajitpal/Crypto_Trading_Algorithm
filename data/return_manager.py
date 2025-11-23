from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple
import math
import numpy as np
import config

class ReturnManager:
    """Tracks rolling prices, returns, and regression features for each symbol."""

    def __init__(
        self,
        regression_window: int,
        risk_window: int,
        feature_mode: str = "log_price",
    ) -> None:
        self.regression_window = max(1, regression_window)
        self.risk_window = max(1, risk_window)
        self.feature_mode = feature_mode

        # Determine buffer sizes with a small safety margin
        self._max_price_window = max(self.regression_window, self.risk_window) + 5
        self._max_return_window = self._max_price_window
        self._max_volume_window = self._max_price_window
        self._max_feature_window = self.regression_window + 5
        self._bar_timestamps: Dict[str, set] = defaultdict(set)

        self.price_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_price_window)
        )
        self.ohlc_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_price_window)
        )
        self.return_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_return_window)
        )
        self.feature_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_feature_window)
        )
        self.volume_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_volume_window)
        )
        self.log_return_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_return_window)
        )

    def update(self, symbol: str, bar: Dict[str, float]) -> None:
        """Ingest the latest validated bar and update rolling statistics."""
        timestamp = bar["timestamp"]
        close = float(bar["close"])
        volume = float(bar.get("volume", 0.0) or 0.0)
        if timestamp in self._bar_timestamps[symbol]:
            # Duplicate bar on grid; skip to avoid double-count
            return
        self._bar_timestamps[symbol].add(timestamp)

        prev_close = self.get_last_close(symbol)
        self.price_history[symbol].append((timestamp, close))
        self.volume_history[symbol].append((timestamp, volume))

        if prev_close and prev_close > 0:
            ret = (close - prev_close) / prev_close
            self.return_history[symbol].append((timestamp, ret))
            log_ret = math.log(close) - math.log(prev_close)
            self.log_return_history[symbol].append((timestamp, log_ret))

        feature_value = math.log(close) if self.feature_mode == "log_price" else close
        self.feature_history[symbol].append((timestamp, feature_value))
        self.ohlc_history[symbol].append(
            (timestamp, float(bar.get("open", close)), float(bar.get("high", close)), float(bar.get("low", close)), close)
        )

    def get_last_close(self, symbol: str) -> Optional[float]:
        """Return the last observed close price."""
        if symbol not in self.price_history or not self.price_history[symbol]:
            return None
        return self.price_history[symbol][-1][1]

    def get_return_series(self, symbol: str, length: Optional[int] = None) -> List[float]:
        """Return the rolling simple returns for the given symbol."""
        series = [value for _, value in self.return_history.get(symbol, [])]
        if length:
            return series[-length:]
        return series

    def get_feature_series(
        self, symbol: str, length: Optional[int] = None
    ) -> List[float]:
        """Return the feature series used by the regression model."""
        series = [value for _, value in self.feature_history.get(symbol, [])]
        if length:
            return series[-length:]
        return series

    def get_price_series(
        self, symbol: str, length: Optional[int] = None
    ) -> List[float]:
        """Return the rolling close prices for diagnostics or regression."""
        series = [value for _, value in self.price_history.get(symbol, [])]
        if length:
            return series[-length:]
        return series

    def get_log_return_series(
        self, symbol: str, length: Optional[int] = None
    ) -> List[float]:
        """Return the rolling log returns for the given symbol."""
        series = [value for _, value in self.log_return_history.get(symbol, [])]
        if length:
            return series[-length:]
        return series

    def get_volume_series(
        self, symbol: str, length: Optional[int] = None
    ) -> List[float]:
        """Return the rolling volumes for the given symbol."""
        series = [value for _, value in self.volume_history.get(symbol, [])]
        if length:
            return series[-length:]
        return series

    def get_return_matrix(
        self, symbols: List[str], window: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Return an aligned matrix of returns for the requested symbols.

        Returns:
            Tuple of (matrix, included_symbols).
        """
        filtered_symbols: List[str] = []
        symbol_returns: List[List[float]] = []

        min_required = window or 1
        for symbol in symbols:
            series = self.get_return_series(symbol)
            if window:
                series = series[-window:]
            if len(series) < min_required:
                continue
            filtered_symbols.append(symbol)
            symbol_returns.append(series)

        if not symbol_returns:
            return np.empty((0, 0)), []

        min_length = min(len(series) for series in symbol_returns)
        aligned = [series[-min_length:] for series in symbol_returns]
        return np.stack(aligned, axis=1), filtered_symbols

    def load_from_candles(self, symbol: str, candles: List[List[float]]) -> None:
        """Replace a symbol's rolling history using raw OHLCV candles."""
        if not candles:
            return

        self.price_history[symbol].clear()
        self.return_history[symbol].clear()
        self.feature_history[symbol].clear()
        self.volume_history[symbol].clear()
        self.log_return_history[symbol].clear()

        for candle in candles:
            if len(candle) < 6:
                continue
            bar = {
                "timestamp": candle[0],
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            }
            self.update(symbol, bar)

    def get_latest_return(self, symbol: str) -> Optional[float]:
        """Return the latest computed simple return."""
        if symbol not in self.return_history or not self.return_history[symbol]:
            return None
        return self.return_history[symbol][-1][1]

    def get_latest_log_return_with_ts(self, symbol: str) -> Optional[tuple]:
        """Return the latest log return with its timestamp."""
        if symbol not in self.log_return_history or not self.log_return_history[symbol]:
            return None
        ts, val = self.log_return_history[symbol][-1]
        return ts, val

    def get_feature_matrix(
        self, symbol: str, window: Optional[int] = None, exclude_outliers: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
        """Construct feature matrix X and target vector y for per-asset forecasting.

        Features include:
            - Lagged log returns (config.LOG_RETURN_LAGS)
            - Lagged volume (config.VOLUME_LAGS)
            - Momentum windows (config.MOMENTUM_WINDOWS)
            - Rolling realized variance (config.VOL_WINDOWS)
            - Rolling high-low range (config.RANGE_WINDOWS)
            - Rolling turnover (config.TURNOVER_WINDOWS)
            - Time-of-day sin/cos (optional)
        Target: next-bar log return.
        """
        log_returns_ts = list(self.log_return_history.get(symbol, []))
        volumes_ts = list(self.volume_history.get(symbol, []))
        prices_ts = list(self.price_history.get(symbol, []))
        ohlc_ts = list(self.ohlc_history.get(symbol, []))

        if not log_returns_ts or not volumes_ts:
            return np.empty((0, 0)), np.empty((0,)), [], []

        lr_len = len(log_returns_ts)
        vol_aligned = volumes_ts[-lr_len:]
        prices_aligned = prices_ts[-(lr_len + 1) :]
        ohlc_aligned = ohlc_ts[-(lr_len + 1) :]

        lags_ret = sorted(config.LOG_RETURN_LAGS)
        lags_vol = sorted(config.VOLUME_LAGS)
        mom_windows = sorted(config.MOMENTUM_WINDOWS)
        vol_windows = sorted(config.VOL_WINDOWS)
        range_windows = sorted(config.RANGE_WINDOWS)
        turnover_windows = sorted(config.TURNOVER_WINDOWS)
        max_lag = max(lags_ret + lags_vol + mom_windows + vol_windows + range_windows + turnover_windows) if (lags_ret or lags_vol or mom_windows or vol_windows or range_windows or turnover_windows) else 0

        effective_window = window or config.REGRESSION_MAX_BARS
        lr_values = [v for _, v in log_returns_ts][-effective_window:]
        lr_ts = [ts for ts, _ in log_returns_ts][-effective_window:]
        vol_values = [v for _, v in vol_aligned][-effective_window:]
        price_values = [p for _, p in prices_aligned][- (effective_window + 1) :]
        range_values = []
        for _, o, h, l, c in ohlc_aligned[-(effective_window + 1) :]:
            range_values.append((h - l) / c if c else 0.0)

        n = len(lr_values)
        if n <= max_lag:
            return np.empty((0, 0)), np.empty((0,)), [], []

        feature_columns: List[str] = []
        feature_columns += [f"ret_lag{lag}" for lag in lags_ret]
        feature_columns += [f"vol_lag{lag}" for lag in lags_vol]
        feature_columns += [f"mom_{w}" for w in mom_windows]
        feature_columns += [f"rv_{w}" for w in vol_windows]
        feature_columns += [f"range_{w}" for w in range_windows]
        feature_columns += [f"turnover_{w}" for w in turnover_windows]
        if config.INCLUDE_TIME_OF_DAY:
            feature_columns += ["tod_sin", "tod_cos"]

        rows: List[List[float]] = []
        targets: List[float] = []
        target_ts: List[int] = []

        for idx in range(max_lag, n):
            row: List[float] = []
            for lag in lags_ret:
                row.append(lr_values[idx - lag])
            for lag in lags_vol:
                row.append(vol_values[idx - lag])
            for w in mom_windows:
                if idx - w >= 0:
                    row.append(sum(lr_values[idx - w : idx]))
                else:
                    row.append(0.0)
            for w in vol_windows:
                if idx - w >= 0:
                    window_slice = lr_values[idx - w : idx]
                    row.append(float(np.mean(np.square(window_slice))))
                else:
                    row.append(0.0)
            for w in range_windows:
                if idx - w >= 0 and len(range_values) >= idx + 1:
                    row.append(float(np.mean(range_values[idx - w : idx])))
                else:
                    row.append(0.0)
            for w in turnover_windows:
                if idx - w >= 0:
                    row.append(float(np.mean(vol_values[idx - w : idx])))
                else:
                    row.append(0.0)
            if config.INCLUDE_TIME_OF_DAY:
                ts_val = lr_ts[idx]
                # ts is ms since epoch
                hour = ((ts_val // 1000) // 3600) % 24
                angle = 2 * np.pi * (hour / 24)
                row.append(np.sin(angle))
                row.append(np.cos(angle))

            target = lr_values[idx]
            rows.append(row)
            targets.append(target)
            target_ts.append(int(lr_ts[idx]))

        if exclude_outliers and rows:
            flagged = set(self.flag_outliers(symbol))
            filtered_rows = []
            filtered_targets = []
            filtered_ts = []
            for r, t, ts_val in zip(rows, targets, target_ts):
                if ts_val in flagged:
                    continue
                filtered_rows.append(r)
                filtered_targets.append(t)
                filtered_ts.append(ts_val)
            rows, targets, target_ts = filtered_rows, filtered_targets, filtered_ts

        X = np.array(rows, dtype=float)
        y = np.array(targets, dtype=float)
        return X, y, target_ts, feature_columns

    def flag_outliers(
        self, symbol: str, sigma_threshold: float = config.OUTLIER_SIGMA_THRESHOLD
    ) -> List[int]:
        """Return timestamps of potential outliers based on raw log-return sigma cap."""
        lr = self.get_log_return_series(symbol)
        if len(lr) < 2:
            return []
        arr = np.array(lr, dtype=float)
        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1) or 1e-9
        flags = []
        for (ts, val) in self.log_return_history.get(symbol, []):
            if abs(val - mu) > sigma_threshold * sigma:
                flags.append(ts)
        return flags
