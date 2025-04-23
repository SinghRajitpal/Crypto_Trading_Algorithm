# Placeholder for the algorithm engine

import asyncio
from typing import Dict, List, Optional
from collections import defaultdict
from .strategies.base_strategy import BaseStrategy
from .trade_signal import TradeSignal
from data.data_engine import DataEngine
import time

class AlgoEngine:
    """Algorithm Engine that processes signals for multiple symbols and timeframes.
    
    This class is responsible for:
    1. Processing market data through trading strategies
    2. Generating trading signals based on strategy logic
    3. Managing signal throttling and deduplication
    4. Orchestrating the trading workflow
    
    Attributes:
        data_engine: Data engine instance for market data access.
        binance_client: Binance client for portfolio data access.
        running: Boolean indicating if the engine is currently running.
        _last_signal_states: Dictionary tracking signal states by symbol.
        _min_signal_interval: Minimum seconds between signals for same symbol.
    """
    
    def __init__(self, data_engine: DataEngine, binance_client):
        """Initializes the Algorithm Engine.
        
        Args:
            data_engine: Data engine instance for market data.
            binance_client: Binance client for portfolio data.
        """
        self.data_engine = data_engine
        self.binance_client = binance_client
        self.running = False
        
        # Signal state tracking
        self._last_signal_states = {}  # {symbol: {timestamp, signal_type, data_hash}}
        self._min_signal_interval = 60  # Minimum seconds between signals for same symbol
        
    def _get_data_hash(self, candles) -> str:
        """Generates a hash of the latest candle data to detect changes.
        
        Args:
            candles: List of candle data.
            
        Returns:
            String hash of the latest candle timestamp and close price.
        """
        if not candles:
            return ""
        latest = candles[-1]
        # Hash the relevant parts of the latest candle
        return f"{latest[0]}_{latest[4]}"  # timestamp_close
        
    def _should_process_signal(self, symbol: str, current_time: int, data_hash: str) -> bool:
        """Determines if we should process a new signal.
        
        Signal processing is based on:
        1. Time since last signal
        2. Whether the data has changed
        
        Args:
            symbol: Trading pair symbol.
            current_time: Current unix timestamp.
            data_hash: Hash of current candle data.
            
        Returns:
            Boolean indicating whether to process a new signal.
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
        """Updates the stored state for a symbol after processing a signal.
        
        Args:
            symbol: Trading pair symbol.
            current_time: Current unix timestamp.
            data_hash: Hash of current candle data.
            signal_type: Type of signal that was processed.
        """
        self._last_signal_states[symbol] = {
            'timestamp': current_time,
            'data_hash': data_hash,
            'signal_type': signal_type
        }
        
    async def process_signals(self, symbol: str, timeframe: str, strategy: BaseStrategy) -> Optional[TradeSignal]:
        """Processes signals for a specific symbol-timeframe pair using the given strategy.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe for the data.
            strategy: Strategy to use for signal generation.
            
        Returns:
            Optional[TradeSignal]: Generated trade signal or None if no signal.
                - Signal types:
                  * open/buy: Enter long position
                  * open/sell: Enter short position  
                  * exit/sell: Exit long position
                  * exit/buy: Exit short position
                  * hold/none: No action needed
                  
        Raises:
            Exception: Any exceptions are caught, logged, and None is returned.
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
                # For hold signals, we still want to return them for logging purposes
                # but the caller might handle them differently
                return signal
                
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to process signals for {symbol}/{timeframe}: {e}")
            return None
            
    async def run(self, strategy: BaseStrategy):
        """Runs the algorithm engine with a single strategy.
        
        This method continuously processes all symbol-timeframe pairs using the
        provided strategy, yielding signals when they are generated.
        
        Args:
            strategy: Strategy to use for signal generation.
            
        Yields:
            TradeSignal: Signal objects with actions "open", "exit", or "hold".
            
        Raises:
            Exception: Any exceptions are caught, logged, and processing continues.
        """
        self.running = True
        
        while self.running:
            try:
                # Process each symbol-timeframe pair
                for symbol, timeframe in self.data_engine.data_fetcher.symbol_timeframes:
                    signal = await self.process_signals(symbol, timeframe, strategy)
                    # Yield all signals including "hold" signals for monitoring
                    if signal:
                        yield signal
                    
                # Sleep to avoid excessive processing
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"[ERROR] Algorithm engine execution failed: {e}")
                await asyncio.sleep(5)  # Sleep longer on error
                
    async def stop(self):
        """Stops the algorithm engine.
        
        Sets the running flag to False, which will cause the run method to exit
        after the current iteration completes.
        """
        self.running = False