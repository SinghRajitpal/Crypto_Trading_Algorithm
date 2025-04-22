# Placeholder for the algorithm engine

import asyncio
from typing import Dict, List, Optional
from collections import defaultdict
from .strategies.base_strategy import BaseStrategy
from .trade_signal import TradeSignal
from data.data_engine import DataEngine
import time

class AlgoEngine:
    """
    Algorithm Engine that processes signals for multiple symbols and timeframes.
    """
    
    def __init__(self, data_engine: DataEngine, binance_client):
        """
        Initialize the Algorithm Engine.
        
        Args:
            data_engine (DataEngine): Data engine instance for market data
            binance_client: Binance client for portfolio data
        """
        self.data_engine = data_engine
        self.binance_client = binance_client
        self.running = False
        
        # Signal state tracking
        self._last_signal_states = {}  # {symbol: {timestamp, signal_type, data_hash}}
        self._min_signal_interval = 60  # Minimum seconds between signals for same symbol
        
    def _get_data_hash(self, candles) -> str:
        """Generate a hash of the latest candle data to detect changes."""
        if not candles:
            return ""
        latest = candles[-1]
        # Hash the relevant parts of the latest candle
        return f"{latest[0]}_{latest[4]}"  # timestamp_close
        
    def _should_process_signal(self, symbol: str, current_time: int, data_hash: str) -> bool:
        """
        Determine if we should process a new signal based on:
        1. Time since last signal
        2. Whether the data has changed
        """
        if symbol not in self._last_signal_states:
            return True
            
        last_state = self._last_signal_states[symbol]
        time_diff = current_time - last_state['timestamp']
        
        # Always process if data has changed
        if data_hash != last_state['data_hash']:
            return True
            
        # Otherwise, only process if enough time has passed
        return time_diff >= self._min_signal_interval
        
    def _update_signal_state(self, symbol: str, current_time: int, data_hash: str, signal_type: str):
        """Update the stored state for a symbol after processing a signal."""
        self._last_signal_states[symbol] = {
            'timestamp': current_time,
            'data_hash': data_hash,
            'signal_type': signal_type
        }
        
    async def process_signals(self, symbol: str, timeframe: str, strategy: BaseStrategy) -> Optional[TradeSignal]:
        """
        Process signals for a specific symbol-timeframe pair using the given strategy.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for the data
            strategy (BaseStrategy): Strategy to use for signal generation
            
        Returns:
            Optional[TradeSignal]: Generated trade signal or None if no signal
        """
        # Get latest candles
        candles = self.data_engine.get_candles(symbol, timeframe)
        
        if not candles:
            return None
            
        try:
            current_time = int(time.time())
            data_hash = self._get_data_hash(candles)
            
            # Check if we should process a new signal
            if not self._should_process_signal(symbol, current_time, data_hash):
                return None
                
            # Calculate signals
            signal = await strategy.calculate_signals(candles, symbol)
            
            if signal:
                # Update state tracking
                self._update_signal_state(symbol, current_time, data_hash, 
                                       f"{signal.action}_{signal.side}")
                return signal
                
            return None
            
        except Exception as e:
            print(f"Error processing signals for {symbol}/{timeframe}: {e}")
            return None
            
    async def run(self, strategy: BaseStrategy):
        """
        Run the algorithm engine with a single strategy.
        
        Args:
            strategy (BaseStrategy): Strategy to use for signal generation
        """
        self.running = True
        
        while self.running:
            try:
                # Process each symbol-timeframe pair
                for symbol, timeframe in self.data_engine.data_fetcher.symbol_timeframes:
                    signal = await self.process_signals(symbol, timeframe, strategy)
                    if signal:
                        yield signal
                    
                # Sleep to avoid excessive processing
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in algorithm engine: {e}")
                await asyncio.sleep(5)  # Sleep longer on error
                
    async def stop(self):
        """
        Stop the algorithm engine.
        """
        self.running = False