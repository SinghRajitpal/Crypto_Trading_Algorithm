from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple
import math
import numpy as np

class ReturnManager:
    """Tracks rolling prices, simple returns, and log-returns for each symbol."""

    def __init__(
        self,
        risk_window: int,
        track_timestamps: bool = True,
    ) -> None:
        self.risk_window = max(1, risk_window)
        self._track_timestamps = track_timestamps

        # Determine buffer sizes with a small safety margin
        self._max_price_window = self.risk_window + 5
        self._max_return_window = self._max_price_window
        self._bar_timestamps: Dict[str, set] = defaultdict(set) if self._track_timestamps else defaultdict(set)

        self.price_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_price_window)
        )
        self.ohlc_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_price_window)
        )
        self.return_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_return_window)
        )
        self.log_return_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_return_window)
        )

    def update(self, symbol: str, bar: Dict[str, float]) -> None:
        """Ingest the latest validated bar and update rolling statistics."""
        timestamp = bar["timestamp"]
        close = float(bar["close"])
        if self._track_timestamps:
            if timestamp in self._bar_timestamps[symbol]:
                # Duplicate bar on grid; skip to avoid double-count
                return
            self._bar_timestamps[symbol].add(timestamp)

        prev_close = self.get_last_close(symbol)
        self.price_history[symbol].append((timestamp, close))

        if prev_close and prev_close > 0:
            ret = (close - prev_close) / prev_close
            self.return_history[symbol].append((timestamp, ret))
            log_ret = math.log(close) - math.log(prev_close)
            self.log_return_history[symbol].append((timestamp, log_ret))
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

    def clear(self) -> None:
        """Release all rolling state to help GC after batch runs."""
        for store in (
            self.price_history,
            self.return_history,
            self.log_return_history,
            self.ohlc_history,
        ):
            store.clear()
        self._bar_timestamps.clear()

    def buffer_lengths(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
        """Return current buffer sizes for diagnostics."""
        symbols = symbols or list(
            set(list(self.price_history.keys()))
            | set(self.return_history.keys())
            | set(self.log_return_history.keys())
            | set(self.ohlc_history.keys())
        )
        snapshot: Dict[str, Dict[str, int]] = {}
        for sym in symbols:
            snapshot[sym] = {
                "price": len(self.price_history.get(sym, [])),
                "returns": len(self.return_history.get(sym, [])),
                "log_returns": len(self.log_return_history.get(sym, [])),
                "ohlc": len(self.ohlc_history.get(sym, [])),
            }
            if self._track_timestamps:
                snapshot[sym]["seen_ts"] = len(self._bar_timestamps.get(sym, []))
        return snapshot
