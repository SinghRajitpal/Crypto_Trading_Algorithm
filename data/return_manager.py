from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple
import math
import numpy as np

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
        self._max_feature_window = self.regression_window + 5

        self.price_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_price_window)
        )
        self.return_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_return_window)
        )
        self.feature_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=self._max_feature_window)
        )

    def update(self, symbol: str, bar: Dict[str, float]) -> None:
        """Ingest the latest validated bar and update rolling statistics."""
        timestamp = bar["timestamp"]
        close = float(bar["close"])

        prev_close = self.get_last_close(symbol)
        self.price_history[symbol].append((timestamp, close))

        if prev_close and prev_close > 0:
            ret = (close - prev_close) / prev_close
            self.return_history[symbol].append((timestamp, ret))

        feature_value = math.log(close) if self.feature_mode == "log_price" else close
        self.feature_history[symbol].append((timestamp, feature_value))

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
