"""
Test data generators for Crypto Trading Algorithm tests.

This module provides utilities for generating test data including
OHLCV candles, market metrics, and statistical calculations.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple


def generate_ohlcv_data(symbol: str, periods: int = 100, 
                       base_price: float = 50000.0, 
                       volatility: float = 0.02,
                       trend: float = 0.0) -> List[List]:
    """Generate realistic OHLCV data for testing.
    
    Args:
        symbol: Trading pair symbol
        periods: Number of candles to generate
        base_price: Starting price
        volatility: Price volatility (standard deviation)
        trend: Trend factor (positive for uptrend, negative for downtrend)
        
    Returns:
        List of OHLCV candles [timestamp, open, high, low, close, volume]
    """
    current_time = int(time.time() * 1000)
    candles = []
    current_price = base_price
    
    for i in range(periods):
        timestamp = current_time - (periods - i) * 60000  # 1-minute candles
        
        # Apply trend
        trend_adjustment = trend * i / periods
        
        # Generate realistic price movement
        price_change = np.random.normal(trend_adjustment, volatility * current_price)
        new_price = max(current_price + price_change, current_price * 0.95)
        
        # Generate OHLC
        open_price = current_price
        close_price = new_price
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.001)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.001)))
        volume = np.random.uniform(100, 1000)
        
        candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
        current_price = new_price
    
    return candles


def calculate_atr(candles: List[List], period: int = 14) -> float:
    """Calculate Average True Range for testing.
    
    Args:
        candles: OHLCV candle data
        period: ATR calculation period
        
    Returns:
        ATR value
    """
    if len(candles) < period + 1:
        return 0.001  # Return minimum ATR
    
    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i][2]
        low = candles[i][3]
        prev_close = candles[i-1][4]
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    # Calculate ATR as simple moving average of true ranges
    if len(true_ranges) >= period:
        return np.mean(true_ranges[-period:])
    else:
        return np.mean(true_ranges) if true_ranges else 0.001


def generate_correlation_matrix(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """Generate correlation matrix for testing.
    
    Args:
        symbols: List of trading symbols
        
    Returns:
        Correlation matrix as nested dictionary
    """
    matrix = {}
    for symbol1 in symbols:
        matrix[symbol1] = {}
        for symbol2 in symbols:
            if symbol1 == symbol2:
                matrix[symbol1][symbol2] = 1.0
            else:
                # Generate random correlation between -0.5 and 0.8
                matrix[symbol1][symbol2] = np.random.uniform(-0.5, 0.8)
    
    return matrix


def generate_volatility_data(symbols: List[str]) -> Dict[str, float]:
    """Generate volatility data for testing.
    
    Args:
        symbols: List of trading symbols
        
    Returns:
        Dictionary mapping symbols to volatility values
    """
    volatility = {}
    for symbol in symbols:
        # Generate realistic volatility between 0.01 and 0.05
        volatility[symbol] = np.random.uniform(0.01, 0.05)
    
    return volatility


def generate_market_data_bar(symbol: str, price: float = 50000.0) -> Dict[str, float]:
    """Generate a single market data bar for testing.
    
    Args:
        symbol: Trading symbol
        price: Base price
        
    Returns:
        Market data bar dictionary
    """
    return {
        'open': price * 0.999,
        'high': price * 1.002,
        'low': price * 0.998,
        'close': price,
        'volume': 100.0
    }


def create_test_signal_metadata(atr_value: float = 0.02, **kwargs) -> Dict[str, Any]:
    """Create test signal metadata.
    
    Args:
        atr_value: ATR value for the signal
        **kwargs: Additional metadata fields
        
    Returns:
        Signal metadata dictionary
    """
    metadata = {
        'atr_value': atr_value,
        'test': True,
        'timestamp': int(time.time() * 1000)
    }
    metadata.update(kwargs)
    return metadata


def generate_price_series(length: int, start_price: float = 50000.0, 
                         volatility: float = 0.02) -> List[float]:
    """Generate a price series for testing.
    
    Args:
        length: Number of price points
        start_price: Starting price
        volatility: Price volatility
        
    Returns:
        List of prices
    """
    prices = [start_price]
    current_price = start_price
    
    for _ in range(length - 1):
        change = np.random.normal(0, volatility * current_price)
        current_price = max(current_price + change, current_price * 0.9)
        prices.append(current_price)
    
    return prices


__all__ = [
    "generate_ohlcv_data",
    "calculate_atr", 
    "generate_correlation_matrix",
    "generate_volatility_data",
    "generate_market_data_bar",
    "create_test_signal_metadata",
    "generate_price_series"
]
