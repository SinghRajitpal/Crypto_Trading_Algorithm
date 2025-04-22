import pandas as pd
import numpy as np
from collections import deque


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

    def get_candle_key(self, symbol, timeframe):
        """Generates a unique key for a symbol-timeframe pair.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Returns:
            String key in the format "symbol_timeframe".
        """
        return f"{symbol}_{timeframe}"

    async def update_tracked_candles(self, symbol, timeframe, latest_candle):
        """Updates the circular buffer with the latest candle data.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            latest_candle: Latest candle data to add.
        """
        key = self.get_candle_key(symbol, timeframe)
        
        # Initialize deque for this symbol-timeframe if it doesn't exist
        if key not in self.symbol_candles:
            self.symbol_candles[key] = deque(maxlen=self.max_candles)
            
        self.symbol_candles[key].append(latest_candle)

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
    
    def get_all_symbols(self):
        """Gets list of all tracked symbol-timeframe pairs.
        
        Returns:
            List of all symbol-timeframe keys currently being tracked.
        """
        return list(self.symbol_candles.keys())

    

    

    
    


 
