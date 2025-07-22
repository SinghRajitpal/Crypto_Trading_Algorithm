"""
Test configuration for Data and Algorithm Engine testing.

This module provides configuration settings, test data, and utilities
specifically for testing the data and algorithm components.
"""

import os
from typing import Dict, List, Any

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Mock configuration for testing
TEST_SYMBOLS = [
    ("BTCUSDT", "1m"),
    ("ETHUSDT", "1m"), 
    ("XRPUSDT", "1m"),
    ("ADAUSDT", "1m"),
]

# Sample candle data for testing
SAMPLE_CANDLES = {
    "BTCUSDT": [
        [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
        [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
        [1642680120000, 42150.0, 42300.0, 42100.0, 42250.0, 110.0],
        [1642680180000, 42250.0, 42400.0, 42200.0, 42350.0, 115.0],
        [1642680240000, 42350.0, 42500.0, 42300.0, 42450.0, 120.0],
    ],
    "ETHUSDT": [
        [1642680000000, 3000.0, 3050.0, 2950.0, 3025.0, 200.0],
        [1642680060000, 3025.0, 3100.0, 3000.0, 3075.0, 210.0],
        [1642680120000, 3075.0, 3150.0, 3050.0, 3125.0, 220.0],
        [1642680180000, 3125.0, 3200.0, 3100.0, 3175.0, 230.0],
        [1642680240000, 3175.0, 3250.0, 3150.0, 3225.0, 240.0],
    ],
    "XRPUSDT": [
        [1642680000000, 0.8000, 0.8100, 0.7900, 0.8050, 1000.0],
        [1642680060000, 0.8050, 0.8200, 0.8000, 0.8150, 1050.0],
        [1642680120000, 0.8150, 0.8300, 0.8100, 0.8250, 1100.0],
        [1642680180000, 0.8250, 0.8400, 0.8200, 0.8350, 1150.0],
        [1642680240000, 0.8350, 0.8500, 0.8300, 0.8450, 1200.0],
    ]
}

# Sample indicator data for testing
SAMPLE_INDICATORS = {
    "sma_5": [42000.0, 42050.0, 42100.0, 42200.0, 42300.0],
    "sma_20": [41900.0, 41950.0, 42000.0, 42050.0, 42100.0],
    "ema_5": [42010.0, 42060.0, 42110.0, 42210.0, 42310.0],
    "ema_20": [41910.0, 41960.0, 42010.0, 42060.0, 42110.0],
    "rsi_14": [45.0, 50.0, 55.0, 60.0, 65.0],
    "atr_14": [50.0, 55.0, 60.0, 65.0, 70.0],
    "bollinger_bands": {
        "upper": [42200.0, 42250.0, 42300.0, 42400.0, 42500.0],
        "middle": [42000.0, 42050.0, 42100.0, 42200.0, 42300.0],
        "lower": [41800.0, 41850.0, 41900.0, 42000.0, 42100.0]
    }
}

# Test parameters for strategies
TEST_STRATEGY_PARAMS = {
    "ma_crossover": {
        "fast_ma_period": 5,
        "slow_ma_period": 20,
        "leverage": 5,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
    },
    "rsi_strategy": {
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
        "leverage": 3,
    },
    "bollinger_strategy": {
        "period": 20,
        "std_multiplier": 2.0,
        "leverage": 4,
    }
}

# Test configuration for different scenarios
TEST_SCENARIOS = {
    "normal_market": {
        "description": "Normal market conditions with regular volatility",
        "candle_count": 100,
        "price_volatility": 0.02,  # 2% max price moves
        "volume_range": (100, 200),
    },
    "volatile_market": {
        "description": "High volatility market conditions",
        "candle_count": 100,
        "price_volatility": 0.05,  # 5% max price moves
        "volume_range": (200, 500),
    },
    "trending_up": {
        "description": "Strong uptrend market",
        "candle_count": 100,
        "trend_direction": "up",
        "trend_strength": 0.001,  # 0.1% per candle
        "price_volatility": 0.02,
    },
    "trending_down": {
        "description": "Strong downtrend market",
        "candle_count": 100,
        "trend_direction": "down",
        "trend_strength": 0.001,  # 0.1% per candle
        "price_volatility": 0.02,
    },
    "sideways": {
        "description": "Sideways/ranging market",
        "candle_count": 100,
        "price_range": (41000.0, 43000.0),
        "price_volatility": 0.015,
    }
}

# Test performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "data_processing": {
        "max_candle_processing_time": 0.001,  # 1ms per candle
        "max_indicator_calculation_time": 0.1,  # 100ms for all indicators
        "max_memory_per_symbol": 10,  # 10MB per symbol-timeframe
    },
    "signal_generation": {
        "max_signal_generation_time": 0.05,  # 50ms per signal
        "max_strategy_calculation_time": 0.02,  # 20ms per strategy
    },
    "integration": {
        "max_end_to_end_latency": 0.2,  # 200ms end-to-end
        "max_concurrent_symbols": 50,  # Support 50 symbols concurrently
    }
}

# Error simulation scenarios for robustness testing
ERROR_SCENARIOS = {
    "network_errors": [
        "Connection timeout",
        "DNS resolution failed",
        "HTTP 503 Service Unavailable",
        "HTTP 429 Rate Limited",
    ],
    "data_errors": [
        "Invalid JSON response",
        "Missing OHLCV data",
        "Timestamp out of order",
        "Negative prices",
        "Zero volume",
    ],
    "calculation_errors": [
        "Division by zero in indicators",
        "NaN values in data",
        "Insufficient data for calculation",
        "Memory allocation error",
    ]
}

# Test utilities configuration
TEST_UTILITIES = {
    "timeout_seconds": 30,  # Default test timeout
    "retry_attempts": 3,     # Retry failed tests
    "parallel_execution": True,  # Allow parallel test execution
    "cleanup_after_tests": True,  # Clean up test artifacts
}

class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_candles(symbol: str, count: int, scenario: str = "normal_market") -> List[List[float]]:
        """Generate synthetic candle data for testing."""
        import random
        import time
        
        scenario_config = TEST_SCENARIOS.get(scenario, TEST_SCENARIOS["normal_market"])
        base_price = 42000.0 if symbol == "BTCUSDT" else 3000.0
        
        candles = []
        current_price = base_price
        timestamp = int(time.time() * 1000)
        
        for i in range(count):
            # Generate OHLCV data based on scenario
            volatility = scenario_config.get("price_volatility", 0.02)
            
            # Price movement
            if scenario == "trending_up":
                trend = scenario_config.get("trend_strength", 0.001) * current_price
            elif scenario == "trending_down":
                trend = -scenario_config.get("trend_strength", 0.001) * current_price
            else:
                trend = 0
            
            # Random price movement
            price_change = random.uniform(-volatility, volatility) * current_price
            new_price = current_price + price_change + trend
            
            # Generate OHLC from current and new price
            open_price = current_price
            close_price = new_price
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.01)
            low_price = min(open_price, close_price) * random.uniform(0.99, 1.0)
            
            # Generate volume
            volume_range = scenario_config.get("volume_range", (100, 200))
            volume = random.uniform(*volume_range)
            
            candle = [
                timestamp + i * 60000,  # 1 minute intervals
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ]
            
            candles.append(candle)
            current_price = close_price
            
        return candles
    
    @staticmethod
    def generate_indicators(candles: List[List[float]], indicators: List[str]) -> Dict[str, Any]:
        """Generate synthetic indicator data for testing."""
        import numpy as np
        
        if not candles:
            return {}
        
        result = {}
        close_prices = [candle[4] for candle in candles]
        
        for indicator in indicators:
            if indicator.startswith("sma_"):
                period = int(indicator.split("_")[1])
                # Simple moving average calculation
                sma_values = []
                for i in range(len(close_prices)):
                    if i < period - 1:
                        sma_values.append(np.nan)
                    else:
                        sma_values.append(np.mean(close_prices[i-period+1:i+1]))
                result[indicator] = np.array(sma_values)
                
            elif indicator.startswith("ema_"):
                period = int(indicator.split("_")[1])
                # Simple EMA approximation
                multiplier = 2 / (period + 1)
                ema_values = [close_prices[0]]
                for i in range(1, len(close_prices)):
                    ema = (close_prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
                    ema_values.append(ema)
                result[indicator] = np.array(ema_values)
                
            elif indicator.startswith("atr_"):
                period = int(indicator.split("_")[1])
                # Simplified ATR calculation
                atr_values = []
                for i in range(len(candles)):
                    if i == 0:
                        atr_values.append(candles[i][2] - candles[i][3])  # High - Low
                    else:
                        true_range = max(
                            candles[i][2] - candles[i][3],  # High - Low
                            abs(candles[i][2] - candles[i-1][4]),  # High - Prev Close
                            abs(candles[i][3] - candles[i-1][4])   # Low - Prev Close
                        )
                        if i < period:
                            atr_values.append(np.mean([atr_values[j] for j in range(i)] + [true_range]))
                        else:
                            # Simple ATR approximation
                            atr_values.append(np.mean([atr_values[j] for j in range(i-period+1, i)] + [true_range]))
                result[indicator] = np.array(atr_values)
        
        return result


class TestValidator:
    """Utility class for validating test results."""
    
    @staticmethod
    def validate_candle_data(candles: List[List[float]]) -> bool:
        """Validate candle data structure and values."""
        if not candles:
            return True  # Empty is valid
        
        for i, candle in enumerate(candles):
            # Check structure
            if len(candle) != 6:
                return False
            
            timestamp, open_price, high, low, close, volume = candle
            
            # Check data types
            if not all(isinstance(x, (int, float)) for x in candle):
                return False
            
            # Check OHLC relationships
            if high < max(open_price, close) or low > min(open_price, close):
                return False
            
            # Check non-negative values
            if any(x < 0 for x in [open_price, high, low, close, volume]):
                return False
            
            # Check timestamp progression
            if i > 0 and timestamp <= candles[i-1][0]:
                return False
        
        return True
    
    @staticmethod
    def validate_signal(signal) -> bool:
        """Validate trade signal structure and content."""
        try:
            # Check required attributes
            required_attrs = ['action', 'side', 'symbol', 'strategy_id', 'metadata', 'signal_confidence']
            if not all(hasattr(signal, attr) for attr in required_attrs):
                return False
            
            # Check valid action values
            if signal.action not in ['open', 'exit', 'hold']:
                return False
            
            # Check valid side values
            if signal.action == 'hold' and signal.side != 'none':
                return False
            elif signal.action != 'hold' and signal.side not in ['buy', 'sell']:
                return False
            
            # Check confidence range
            if not 0.0 <= signal.signal_confidence <= 1.0:
                return False
            
            return True
            
        except AttributeError:
            return False


# Export key components
__all__ = [
    'TEST_SYMBOLS',
    'SAMPLE_CANDLES', 
    'SAMPLE_INDICATORS',
    'TEST_STRATEGY_PARAMS',
    'TEST_SCENARIOS',
    'PERFORMANCE_BENCHMARKS',
    'ERROR_SCENARIOS',
    'TestDataGenerator',
    'TestValidator'
]
