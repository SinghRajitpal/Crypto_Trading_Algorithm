# Placeholder for the algorithm engine

import asyncio
from typing import Dict, Optional
import config
from .strategies.base_strategy import BaseStrategy
from .trade_signal import TradeSignal
from data.data_engine import DataEngine
import time
from utils.logging_config import get_logger, console_log

# Get logger for this module
logger = get_logger(__name__)

class AlgoEngine:
    """Algorithm Engine that processes signals for multiple symbols and timeframes.
    
    This class is responsible for:
    1. Processing market data through trading strategies
    2. Generating trading signals based on strategy logic
    3. Managing signal throttling and deduplication
    4. Orchestrating the trading workflow
    
    Attributes:
        data_engine: Data engine instance for market data access.
        running: Boolean indicating if the engine is currently running.
        _last_signal_states: Dictionary tracking signal states by symbol.
        _min_signal_interval: Minimum seconds between signals for same symbol.
    """
    
    def __init__(self, data_engine: DataEngine):
        """Initializes the Algorithm Engine.
        
        Args:
            data_engine: Data engine instance for market data.
        """
        self.data_engine = data_engine
        self.running = False
        
        # Get the binance_client from data_engine
        self.binance_client = data_engine.binance_client if hasattr(data_engine, 'binance_client') else None
        
        # Track last signal per *symbol–timeframe* so that we can run multiple
        # timeframes per symbol without collisions (e.g. BTC 5m vs BTC 1h).
        #
        # Key format: ``f"{symbol}_{timeframe}"``.
        self._last_signal_states = {}  # {sym_tf_key: {timestamp, signal_type, data_hash}}
        self._min_signal_interval = 60  # Minimum seconds between signals for same symbol
        
        logger.info("AlgoEngine initialized")
        
    def _get_data_hash(self, candles) -> str:
        """Generates a hash of the latest candle data to detect changes.
        
        Args:
            candles: List of candle data.
            
        Returns:
            String hash of the latest candle timestamp and close price.
        """
        if not candles or len(candles) == 0:
            return ""
        latest = candles[-1]
        # Hash the relevant parts of the latest candle
        return f"{latest[0]}_{latest[4]}"  # timestamp_close
        
    def _should_process_signal(self, key: str, current_time: int, data_hash: str) -> bool:
        """Determines if we should process a new signal.
        
        Signal processing is based on:
        1. Time since last signal
        2. Whether the data has changed
        
        Args:
            key: Symbol-timeframe key.
            current_time: Current unix timestamp.
            data_hash: Hash of current candle data.
            
        Returns:
            Boolean indicating whether to process a new signal.
        """
        if key not in self._last_signal_states:
            return True
            
        last_state = self._last_signal_states[key]
        time_diff = current_time - last_state['timestamp']
        
        # Always process if data has changed
        if data_hash != last_state['data_hash']:
            return True
            
        # Otherwise, only process if enough time has passed
        return time_diff >= self._min_signal_interval
        
    def _update_signal_state(self, key: str, current_time: int, data_hash: str, signal_type: str):
        """Updates the stored state for a symbol after processing a signal.
        
        Args:
            key: Symbol-timeframe key.
            current_time: Current unix timestamp.
            data_hash: Hash of current candle data.
            signal_type: Type of signal that was processed.
        """
        self._last_signal_states[key] = {
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
        try:
            # Get latest candles
            candles = self.data_engine.get_candles(symbol, timeframe)
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}_{timeframe}: {e}")
            return None
        
        if not candles or len(candles) == 0:
            return None
            
        try:
            current_time = int(time.time())
            data_hash = self._get_data_hash(candles)
            
            # Use combined key to track per symbol-timeframe pair
            key = f"{symbol}_{timeframe}"

            # Check if we should process a new signal
            if not self._should_process_signal(key, current_time, data_hash):
                return None
                
            # Calculate signals
            signal = await strategy.calculate_signals(candles, symbol)
            
            if signal:
                # Update signal state
                self._update_signal_state(
                    key=key,
                    current_time=current_time,
                    data_hash=data_hash,
                    signal_type=f"{signal.action}/{signal.side}"
                )
                
                # Set timestamp if not already set
                if not signal.timestamp:
                    signal.timestamp = current_time * 1000  # Convert to milliseconds for consistency
                
                logger.info(f"Generated {signal.action}/{signal.side} signal for {symbol} with confidence {signal.signal_confidence:.2f}")
                
            return signal
            
        except Exception as e:
            logger.error(f"Error processing signals for {symbol}/{timeframe}: {e}")
            return None
    
    async def run(self, strategy: BaseStrategy):
        """Run the algorithm engine with the specified strategy.
        
        This method continuously processes signals for all configured
        symbol-timeframe pairs in the strategy.
        
        Args:
            strategy: Trading strategy to use.
            
        Yields:
            Generated trade signals as they are calculated.
        """
        if not hasattr(self, 'running') or not self.running:
            self.running = True
            logger.info(f"Starting with strategy: {strategy.strategy_id}")
        
        while self.running:
            try:
                for symbol, timeframe in config.symbols:
                    signal = await self.process_signals(symbol, timeframe, strategy)
                    
                    if signal:
                        yield signal
                        
                # Throttle processing speed
                await asyncio.sleep(1)  # Check for new signals every second
                
            except Exception as e:
                logger.error(f"Error in main processing loop: {e}")
                await asyncio.sleep(5)  # Sleep longer on error
    
    async def stop(self):
        """Stop the algorithm engine."""
        self.running = False
        logger.info("AlgoEngine stopped")