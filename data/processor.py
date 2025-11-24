import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, List

import config
from data.duplicate_tracker import DuplicateTracker


class DataProcessor:
    """Data Processor for storing and managing market data.
    
    This class provides a circular buffer implementation for storing
    candle data for multiple symbol-timeframe pairs and methods to
    access that data.
    
    Attributes:
        max_candles: Maximum number of candles to store per symbol-timeframe.
        symbol_candles: Dictionary of candle data for each symbol-timeframe pair.
    """
    
    def __init__(self, max_candles):
        """Initializes the Data Processor.
        
        Args:
            max_candles: Maximum number of candles to store in memory per symbol-timeframe.
        """
        self.max_candles = max_candles
        self.symbol_candles = {}  # Dictionary to hold deques for each symbol-timeframe pair
        self._last_timestamp: Dict[str, int] = {}
        self._missing_bars: Dict[str, List[int]] = {}
        self._dup_tracker = DuplicateTracker()

    def get_candle_key(self, symbol, timeframe):
        """Generates a unique key for a symbol-timeframe pair.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Returns:
            String key in the format "symbol_timeframe".
        """
        return f"{symbol}_{timeframe}"

    @staticmethod
    def _timeframe_ms(timeframe: str) -> int:
        """Convert timeframe like '5m' or '1h' to milliseconds."""
        if timeframe.endswith("m"):
            mins = int(timeframe[:-1])
        elif timeframe.endswith("h"):
            mins = int(timeframe[:-1]) * 60
        elif timeframe.endswith("d"):
            mins = int(timeframe[:-1]) * 60 * 24
        else:
            raise ValueError("Timeframe must end with m/h/d for grid alignment")
        return mins * 60 * 1000

    @staticmethod
    def _align_to_grid(timestamp_ms: int, timeframe: str = config.BAR_GRID_TIMEFRAME) -> int:
        """Align a timestamp to the start of the bar grid."""
        return (timestamp_ms // DataProcessor._timeframe_ms(timeframe)) * DataProcessor._timeframe_ms(timeframe)

    async def update_tracked_candles(self, symbol, timeframe, latest_candle):
        """Updates the circular buffer with the latest candle data.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            latest_candle: Latest candle data to add.
        """
        key = self.get_candle_key(symbol, timeframe)
        tf_ms = self._timeframe_ms(timeframe)

        # Align timestamp to global grid
        if latest_candle and len(latest_candle) > 0:
            latest_candle = list(latest_candle)
            latest_candle[0] = self._align_to_grid(int(latest_candle[0]), timeframe)

        # Initialize deque for this symbol-timeframe if it doesn't exist
        if key not in self.symbol_candles:
            self.symbol_candles[key] = deque(maxlen=self.max_candles)
            self._missing_bars[key] = []
            
        # Skip duplicate or out-of-order timestamps on the grid
        last_ts = self._last_timestamp.get(key)
        current_ts = int(latest_candle[0]) if latest_candle and len(latest_candle) > 0 else None
        if current_ts is None:
            return
        # Cross-symbol duplicate detection per timeframe
        if self._dup_tracker.seen_before(timeframe, current_ts):
            # Already ingested this grid timestamp for another symbol; still store but can log upstream
            pass
        if last_ts is not None:
            if current_ts == last_ts:
                return
            if current_ts < last_ts:
                return
            # Detect missing bars on the grid
            expected = last_ts + tf_ms
            while expected < current_ts:
                self._missing_bars[key].append(expected)
                expected += tf_ms

        self.symbol_candles[key].append(latest_candle)
        self._last_timestamp[key] = current_ts

    def get_candles(self, symbol, timeframe):
        """Gets all currently tracked candles for a specific symbol-timeframe pair.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Returns:
            List of candles for the specified symbol and timeframe, or empty list if none.
        """
        key = self.get_candle_key(symbol, timeframe)
        if key in self.symbol_candles:
            return list(self.symbol_candles[key])
        return []

    def get_latest_candle(self, symbol, timeframe):
        """Gets the most recent candle for a specific symbol-timeframe pair.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Returns:
            Most recent candle or None if no candles are available.
        """
        key = self.get_candle_key(symbol, timeframe)
        if key in self.symbol_candles and self.symbol_candles[key]:
            return self.symbol_candles[key][-1]
        return None
    
    def get_missing_bars(self, symbol: str, timeframe: str) -> List[int]:
        """Return list of missing grid timestamps for this symbol-timeframe."""
        key = self.get_candle_key(symbol, timeframe)
        return list(self._missing_bars.get(key, []))
    
    def get_all_symbols(self):
        """Gets list of all tracked symbol-timeframe pairs.
        
        Returns:
            List of all symbol-timeframe keys currently being tracked.
        """
        return list(self.symbol_candles.keys())

    

    

    
    


 
