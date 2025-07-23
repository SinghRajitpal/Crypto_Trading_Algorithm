"""
Mock objects for testing the Crypto Trading Algorithm.

This module provides comprehensive mock implementations for testing without
external dependencies or real market data.
"""

import time
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import Mock
from algorithm.strategies.base_strategy import BaseStrategy
from algorithm.trade_signal import TradeSignal


class MockDataEngine:
    """Mock data engine for testing purposes."""
    
    def __init__(self):
        """Initialize mock data engine."""
        self.call_count = {}
        self.binance_client = None
        self.candles_data = {}  # Add candles_data attribute for test compatibility
        
    def get_candles(self, symbol: str, timeframe: str) -> List[List]:
        """Generate mock candle data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for the data
            
        Returns:
            List of OHLCV candles
        """
        # Check if test data is available first
        test_key = (symbol, timeframe)
        if test_key in self.candles_data:
            return self.candles_data[test_key]
        
        # Track calls for different behavior
        key = f"{symbol}_{timeframe}"
        self.call_count[key] = self.call_count.get(key, 0) + 1
        
        # Generate slightly different data each call
        current_time = int(time.time() * 1000)
        offset = self.call_count[key] * 60000  # 1 minute offset per call
        
        # Set base price based on symbol
        if 'BTC' in symbol:
            base_price = 50000.0
        elif 'ETH' in symbol:
            base_price = 3000.0
        else:
            base_price = 0.5
        
        candles = []
        for i in range(20):
            timestamp = current_time - offset + (i * 60000)
            price = base_price + (i * 10)  # Slight upward trend
            
            candles.append([
                timestamp,
                price,         # open
                price + 5,     # high
                price - 5,     # low
                price + 2,     # close
                100.0         # volume
            ])
        
        return candles
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> List:
        """Get the latest candle for a symbol."""
        candles = self.get_candles(symbol, timeframe)
        return candles[-1] if candles else None


class MockBinanceClient:
    """Mock Binance client for testing execution engine."""
    
    def __init__(self):
        """Initialize mock Binance client."""
        self.account_config_called = False
        self.orders = []
        self.positions = []
        self.balance = {'USDT': 10000.0}
        self.order_id_counter = 0
        
    async def setup_account_config(self):
        """Mock account configuration setup."""
        self.account_config_called = True
        
    async def get_balance(self):
        """Mock balance retrieval."""
        return {
            'total': self.balance,
            'free': self.balance,
            'used': {}
        }
        
    async def get_open_positions(self, symbol=None):
        """Mock open positions retrieval."""
        if symbol:
            return [p for p in self.positions if p['symbol'] == symbol]
        return self.positions
        
    async def open_position(self, symbol, side, amount, price=None, 
                          stop_loss=None, take_profit=None, leverage=None, margin_type=None):
        """Mock position opening."""
        self.order_id_counter += 1
        
        position = {
            'id': f'position_{self.order_id_counter}',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'contracts': amount,
            'entry_price': price or 50000.0,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'leverage': leverage or 1,
            'margin_type': margin_type or 'isolated',
            'timestamp': int(time.time() * 1000),
            'status': 'open'
        }
        
        self.positions.append(position)
        
        order = {
            'id': f'order_{self.order_id_counter}',
            'symbol': symbol,
            'type': 'market',
            'side': side,
            'amount': amount,
            'price': price or 50000.0,
            'status': 'filled',
            'timestamp': int(time.time() * 1000)
        }
        self.orders.append(order)
        
        return {'status': 'success', 'order': order, 'position': position}


class MockStrategy(BaseStrategy):
    """Mock strategy for testing purposes."""
    
    def __init__(self, signal_sequence: List[str] = None, strategy_id: str = "test_strategy"):
        """Initialize mock strategy.
        
        Args:
            signal_sequence: Sequence of signals to generate ('buy', 'sell', 'hold')
            strategy_id: Strategy identifier
        """
        super().__init__(params={}, strategy_id=strategy_id)
        self.signal_sequence = signal_sequence or ["hold"]
        self.signal_index = 0
    
    def get_required_indicators(self) -> List[str]:
        """Return empty list for testing."""
        return []
    
    def _generate_signals(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Generate test signals in sequence."""
        signal_type = self.signal_sequence[self.signal_index % len(self.signal_sequence)]
        self.signal_index += 1
        
        if signal_type == "buy":
            return {"action": "open", "side": "buy", "confidence": 0.8}
        elif signal_type == "sell":
            return {"action": "open", "side": "sell", "confidence": 0.8}
        elif signal_type == "exit":
            return {"action": "exit", "side": "sell", "confidence": 0.8}
        else:
            return {"action": "hold", "side": "none", "confidence": 0.5}
    
    async def calculate_signals(self, candles, symbol: str):
        """Override to return TradeSignal."""
        result = self._generate_signals(symbol, {})
        
        return TradeSignal(
            action=result["action"],
            side=result["side"],
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata={"test": True, "atr_value": 0.02},
            signal_confidence=result["confidence"]
        )


class MockErrorStrategy(BaseStrategy):
    """Strategy that throws errors for testing error handling."""
    
    def __init__(self, strategy_id: str = "error_strategy"):
        """Initialize error strategy."""
        super().__init__(params={}, strategy_id=strategy_id)
    
    def get_required_indicators(self) -> List[str]:
        """Return empty list."""
        return []
    
    def _generate_signals(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Raise error for testing."""
        raise Exception("Intentional strategy error")
    
    async def calculate_signals(self, candles, symbol: str):
        """Raise error for testing."""
        raise Exception("Intentional strategy error")


class MockDataEngineWithTrend:
    """Mock data engine that generates trending data for integration tests."""
    
    def __init__(self):
        """Initialize mock data engine with trend generation."""
        self.candle_data = {}
        self.binance_client = None
        
    def get_candles(self, symbol: str, timeframe: str) -> List[List]:
        """Generate trending mock candle data."""
        key = f"{symbol}_{timeframe}"
        if key not in self.candle_data:
            self.candle_data[key] = self._generate_trending_data(symbol, 50)
        return self.candle_data[key]
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> List:
        """Get the latest candle."""
        candles = self.get_candles(symbol, timeframe)
        return candles[-1] if candles else None
    
    def _generate_trending_data(self, symbol: str, periods: int) -> List[List]:
        """Generate trending OHLCV data for signal generation."""
        current_time = int(time.time() * 1000)
        candles = []
        
        # Set base price based on symbol
        if 'BTC' in symbol:
            base_price = 50000.0
        elif 'ETH' in symbol:
            base_price = 3000.0
        else:
            base_price = 0.5
        
        for i in range(periods):
            timestamp = current_time - (periods - i) * 60000
            # Create trending pattern
            trend_factor = 1 + (0.002 * i)  # Gradual uptrend
            price = base_price * trend_factor
            
            candles.append([
                timestamp,
                price * 0.999,  # open
                price * 1.002,  # high
                price * 0.998,  # low
                price,          # close
                100.0          # volume
            ])
        
        return candles


__all__ = [
    "MockDataEngine",
    "MockBinanceClient", 
    "MockStrategy",
    "MockErrorStrategy",
    "MockDataEngineWithTrend"
]
