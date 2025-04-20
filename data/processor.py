import pandas as pd
import numpy as np
from collections import deque


class DataProcessor:
    def __init__(self, max_candles):
        self.max_candles = max_candles
        self.symbol_candles = {}  # Dictionary to hold deques for each symbol-timeframe pair

    def get_candle_key(self, symbol, timeframe):
        """Generate a unique key for a symbol-timeframe pair"""
        return f"{symbol}_{timeframe}"

    async def update_tracked_candles(self, symbol, timeframe, latest_candle):
        """Update the circular buffer with the latest candle data for a specific symbol-timeframe pair"""
        key = self.get_candle_key(symbol, timeframe)
        
        # Initialize deque for this symbol-timeframe if it doesn't exist
        if key not in self.symbol_candles:
            self.symbol_candles[key] = deque(maxlen=self.max_candles)
            
        self.symbol_candles[key].append(latest_candle)

    def get_candles(self, symbol, timeframe):
        """Get all currently tracked candles for a specific symbol-timeframe pair"""
        key = self.get_candle_key(symbol, timeframe)
        if key in self.symbol_candles:
            return list(self.symbol_candles[key])
        return []

    def get_latest_candle(self, symbol, timeframe):
        """Get the most recent candle for a specific symbol-timeframe pair"""
        key = self.get_candle_key(symbol, timeframe)
        if key in self.symbol_candles and self.symbol_candles[key]:
            return self.symbol_candles[key][-1]
        return None
    
    def get_all_symbols(self):
        """Get list of all tracked symbol-timeframe pairs"""
        return list(self.symbol_candles.keys())

    

    

    
    


 
