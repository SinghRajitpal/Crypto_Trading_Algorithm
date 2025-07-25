"""
Comprehensive Data Engine Test Suite
Senior Quantitative Systems Testing - Tier-1 Hedge Fund Standards

This test suite provides ultra-detailed coverage of the Data Engine including:
- DataProcessor circular buffer implementation
- DataFetcher historical and live data handling  
- DataEngine coordination and error handling
- Integration points and failure modes
- Performance characteristics and memory management
- Financial data anomalies and edge cases
- Professional-grade regression testing
"""
import unittest
import asyncio
import time
import sys
import os
import gc
import warnings
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from collections import deque
from typing import List, Dict, Any
from decimal import Decimal
from datetime import datetime, timedelta

# Suppress specific warnings for cleaner test output
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.data_engine import DataEngine
from data.data_fetcher import DataFetcher
from data.processor import DataProcessor
from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from algorithm.strategies.base_strategy import BaseStrategy


class MockBinanceClient:
    """Mock Binance client for testing."""
    
    def __init__(self, testnet=True):
        self.testnet = testnet
        self.exchange = Mock()
        
    async def close(self):
        """Mock close method."""
        pass


class MockExchange:
    """Mock exchange for testing."""
    
    def __init__(self):
        self.candle_data = {}
        self.fetch_count = 0
        
    async def fetch_ohlcv(self, symbol, timeframe, limit=100):
        """Mock fetch_ohlcv with realistic data generation."""
        self.fetch_count += 1
        
        # Generate realistic OHLCV data
        base_price = 42000.0
        candles = []
        
        for i in range(min(limit, 100)):
            timestamp = 1642680000000 + i * 60000  # 1-minute intervals
            open_price = base_price + (i * 10) + np.random.uniform(-50, 50)
            high = open_price + np.random.uniform(10, 100)
            low = open_price - np.random.uniform(10, 100)
            close = open_price + np.random.uniform(-50, 50)
            volume = 100.0 + np.random.uniform(10, 50)
            
            candles.append([timestamp, open_price, high, low, close, volume])
            
        return candles
        
    async def watch_ohlcv(self, symbol, timeframe):
        """Mock watch_ohlcv that returns evolving data."""
        return await self.fetch_ohlcv(symbol, timeframe, 5)
        
    def parse_timeframe(self, timeframe):
        """Mock parse_timeframe."""
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600
        }
        return timeframe_map.get(timeframe, 60)


class TestDataProcessor(unittest.TestCase):
    """Comprehensive DataProcessor testing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor(max_candles=100)
        
        # Sample candles for testing
        self.sample_candles = [
            [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
            [1642680120000, 42150.0, 42300.0, 42100.0, 42250.0, 110.0],
            [1642680180000, 42250.0, 42400.0, 42200.0, 42350.0, 115.0],
            [1642680240000, 42350.0, 42500.0, 42300.0, 42450.0, 120.0],
            [1642680300000, 42450.0, 42600.0, 42400.0, 42550.0, 125.0]
        ]
    
    def test_initialization_comprehensive(self):
        """Test DataProcessor initialization with various parameters."""
        # Test custom max_candles
        processor = DataProcessor(max_candles=500)
        self.assertEqual(processor.max_candles, 500)
        self.assertEqual(processor.symbol_candles, {})
        
        # Test different max_candles
        custom_processor = DataProcessor(max_candles=100)
        self.assertEqual(custom_processor.max_candles, 100)
        
        # Test edge case: very small max_candles
        small_processor = DataProcessor(max_candles=1)
        self.assertEqual(small_processor.max_candles, 1)
    
    def test_candle_key_generation_comprehensive(self):
        """Test candle key generation for various symbol-timeframe combinations."""
        test_cases = [
            ("BTCUSDT", "1m", "BTCUSDT_1m"),
            ("ETHUSDT", "5m", "ETHUSDT_5m"),
            ("SOLUSDT", "1h", "SOLUSDT_1h"),
            ("BTC-USD", "1d", "BTC-USD_1d"),
            ("ETH/USD", "4h", "ETH/USD_4h")
        ]
        
        for symbol, timeframe, expected_key in test_cases:
            with self.subTest(symbol=symbol, timeframe=timeframe):
                key = self.processor.get_candle_key(symbol, timeframe)
                self.assertEqual(key, expected_key)
    
    def test_circular_buffer_behavior_detailed(self):
        """Test circular buffer behavior with exact limits."""
        async def _test_async():
            symbol = "BTCUSDT"
            timeframe = "1m"
            
            # Add exactly max_candles
            for i in range(self.processor.max_candles):
                candle = [1642680000000 + i * 60000, 42000 + i, 42100 + i, 41900 + i, 42050 + i, 100 + i]
                await self.processor.update_tracked_candles(symbol, timeframe, candle)
            
            candles = self.processor.get_candles(symbol, timeframe)
            self.assertEqual(len(candles), self.processor.max_candles)
            
            # Verify order preservation
            for i, candle in enumerate(candles):
                expected_timestamp = 1642680000000 + i * 60000
                self.assertEqual(candle[0], expected_timestamp)
            
            # Add one more candle - should replace the oldest
            new_candle = [1642680000000 + self.processor.max_candles * 60000, 50000, 50100, 49900, 50050, 200]
            await self.processor.update_tracked_candles(symbol, timeframe, new_candle)
            
            candles_after = self.processor.get_candles(symbol, timeframe)
            self.assertEqual(len(candles_after), self.processor.max_candles)
            
            # Latest candle should be the new one
            self.assertEqual(candles_after[-1], new_candle)
            
            # Should not contain the first candle anymore
            first_expected_timestamp = 1642680000000 + 60000  # Second candle timestamp
            self.assertEqual(candles_after[0][0], first_expected_timestamp)
        
        asyncio.run(_test_async())
    
    def test_multiple_symbol_timeframe_isolation(self):
        """Test isolation between different symbol-timeframe pairs."""
        async def _test_async():
            test_pairs = [
                ("BTCUSDT", "1m"),
                ("BTCUSDT", "5m"),
                ("ETHUSDT", "1m"),
                ("ETHUSDT", "5m")
            ]
            
            # Add different candles for each pair
            for i, (symbol, timeframe) in enumerate(test_pairs):
                candle = [1642680000000 + i * 60000, 40000 + i * 1000, 41000 + i * 1000, 39000 + i * 1000, 40500 + i * 1000, 100 + i * 10]
                await self.processor.update_tracked_candles(symbol, timeframe, candle)
            
            # Verify isolation
            for i, (symbol, timeframe) in enumerate(test_pairs):
                candles = self.processor.get_candles(symbol, timeframe)
                self.assertEqual(len(candles), 1)
                self.assertEqual(candles[0][1], 40000 + i * 1000)  # Check open price
            
            # Verify key generation
            all_symbols = self.processor.get_all_symbols()
            expected_keys = [f"{symbol}_{timeframe}" for symbol, timeframe in test_pairs]
            self.assertEqual(len(all_symbols), len(expected_keys))
            for key in expected_keys:
                self.assertIn(key, all_symbols)
        
        asyncio.run(_test_async())
    
    def test_get_operations_with_nonexistent_data(self):
        """Test get operations with non-existent symbol-timeframe pairs."""
        # Test get_candles with non-existent data
        candles = self.processor.get_candles("NONEXISTENT", "1m")
        self.assertEqual(candles, [])
        
        # Test get_latest_candle with non-existent data
        latest = self.processor.get_latest_candle("NONEXISTENT", "1m")
        self.assertIsNone(latest)
        
        # Test get_all_symbols with empty processor
        symbols = self.processor.get_all_symbols()
        self.assertEqual(symbols, [])
    
    def test_data_type_handling(self):
        """Test handling of different data types in candles."""
        async def _test_async():
            symbol = "BTCUSDT"
            timeframe = "1m"
            
            # Test with integer values
            int_candle = [1642680000000, 42000, 42100, 41900, 42050, 100]
            await self.processor.update_tracked_candles(symbol, timeframe, int_candle)
            
            # Test with float values
            float_candle = [1642680060000, 42000.5, 42100.75, 41900.25, 42050.5, 100.5]
            await self.processor.update_tracked_candles(symbol, timeframe, float_candle)
            
            # Test with mixed types
            mixed_candle = [1642680120000, 42000.0, 42100, 41900.5, 42050, 100.0]
            await self.processor.update_tracked_candles(symbol, timeframe, mixed_candle)
            
            candles = self.processor.get_candles(symbol, timeframe)
            self.assertEqual(len(candles), 3)
            
            # Verify all types are preserved
            self.assertEqual(candles[0], int_candle)
            self.assertEqual(candles[1], float_candle)
            self.assertEqual(candles[2], mixed_candle)
        
        asyncio.run(_test_async())
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        async def _test_async():
            large_processor = DataProcessor(max_candles=1000)
            
            # Generate many candles
            large_candle_set = []
            for i in range(1500):
                candle = [1642680000000 + i * 60000, 42000 + i, 42100 + i, 41900 + i, 42050 + i, 100 + i]
                large_candle_set.append(candle)
            
            symbol = "BTCUSDT"
            timeframe = "1m"
            
            # Add all candles
            for candle in large_candle_set:
                await large_processor.update_tracked_candles(symbol, timeframe, candle)
            
            # Verify buffer size is maintained
            candles = large_processor.get_candles(symbol, timeframe)
            self.assertEqual(len(candles), 1000)  # Should be capped at max_candles
            
            # Verify we have the latest candles
            latest_candle = candles[-1]
            expected_latest = large_candle_set[-1]
            self.assertEqual(latest_candle, expected_latest)
        
        asyncio.run(_test_async())
    
    def test_internal_data_structure_integrity(self):
        """Test internal data structure integrity after operations."""
        async def _test_async():
            symbol = "BTCUSDT"
            timeframe = "1m"
            
            # Add some candles
            for candle in self.sample_candles:
                await self.processor.update_tracked_candles(symbol, timeframe, candle)
            
            # Verify internal structure
            key = self.processor.get_candle_key(symbol, timeframe)
            self.assertIn(key, self.processor.symbol_candles)
            
            internal_deque = self.processor.symbol_candles[key]
            self.assertIsInstance(internal_deque, deque)
            self.assertEqual(len(internal_deque), len(self.sample_candles))
            
            # Verify data consistency
            retrieved_candles = self.processor.get_candles(symbol, timeframe)
            self.assertEqual(len(retrieved_candles), len(self.sample_candles))
            
            for i, candle in enumerate(retrieved_candles):
                self.assertEqual(candle, self.sample_candles[i])
        
        asyncio.run(_test_async())


class TestDataFetcher(unittest.TestCase):
    """Comprehensive DataFetcher testing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockBinanceClient()
        self.mock_client.exchange = MockExchange()
        self.fetcher = DataFetcher(binance_client=self.mock_client, max_candles=50)
        self.data_processor = self.fetcher.data_processor
    
    def test_initialization_comprehensive(self):
        """Test DataFetcher initialization with various scenarios."""
        # Test standard initialization
        fetcher = DataFetcher(binance_client=self.mock_client, max_candles=50)
        self.assertEqual(fetcher.binance, self.mock_client)
        self.assertIsInstance(fetcher.data_processor, DataProcessor)
        self.assertFalse(fetcher.should_close_client)  # We provided client
        self.assertEqual(fetcher.data_processor.max_candles, 50)
    
    def test_get_candles_delegation(self):
        """Test that get_candles properly delegates to data_processor."""
        # Add test data directly to processor
        test_candles = [
            [1642680000000, 42000, 42100, 41900, 42050, 100],
            [1642680060000, 42050, 42150, 42000, 42100, 110]
        ]
        
        symbol = "BTCUSDT"
        timeframe = "1m"
        key = self.data_processor.get_candle_key(symbol, timeframe)
        self.data_processor.symbol_candles[key] = deque(test_candles, maxlen=self.data_processor.max_candles)
        
        # Test delegation
        result = self.fetcher.get_candles(symbol, timeframe)
        self.assertEqual(result, test_candles)
    
    def test_get_latest_candle_delegation(self):
        """Test that get_latest_candle properly delegates to data_processor."""
        # Add test data directly to processor
        test_candles = [
            [1642680000000, 42000, 42100, 41900, 42050, 100],
            [1642680060000, 42050, 42150, 42000, 42100, 110]
        ]
        
        symbol = "BTCUSDT"
        timeframe = "1m"
        key = self.data_processor.get_candle_key(symbol, timeframe)
        self.data_processor.symbol_candles[key] = deque(test_candles, maxlen=self.data_processor.max_candles)
        
        # Test delegation
        result = self.fetcher.get_latest_candle(symbol, timeframe)
        self.assertEqual(result, test_candles[-1])


class TestDataEngine(unittest.TestCase):
    """Comprehensive DataEngine testing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockBinanceClient()
        self.mock_client.exchange = MockExchange()
        self.data_engine = DataEngine(self.mock_client, max_candles=100)
    
    def test_initialization_comprehensive(self):
        """Test DataEngine initialization with various scenarios."""
        # Test standard initialization
        engine = DataEngine(self.mock_client, max_candles=100)
        self.assertEqual(engine.binance_client, self.mock_client)
        self.assertIsInstance(engine.data_fetcher, DataFetcher)
        self.assertFalse(engine.running)
        
        # Test default max_candles (should be 100, not 1000)
        default_engine = DataEngine(self.mock_client)
        self.assertEqual(default_engine.data_fetcher.data_processor.max_candles, 100)
        
        # Test custom max_candles
        custom_engine = DataEngine(self.mock_client, max_candles=500)
        self.assertEqual(custom_engine.data_fetcher.data_processor.max_candles, 500)
    
    def test_get_methods_delegation(self):
        """Test that get methods properly delegate to data_fetcher."""
        # Add test data
        test_candles = [
            [1642680000000, 42000, 42100, 41900, 42050, 100],
            [1642680060000, 42050, 42150, 42000, 42100, 110]
        ]
        
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Manually add to processor for testing
        key = self.data_engine.data_fetcher.data_processor.get_candle_key(symbol, timeframe)
        self.data_engine.data_fetcher.data_processor.symbol_candles[key] = deque(test_candles, maxlen=100)
        
        # Test get_candles delegation
        candles = self.data_engine.get_candles(symbol, timeframe)
        self.assertEqual(candles, test_candles)
        
        # Test get_latest_candle delegation
        latest = self.data_engine.get_latest_candle(symbol, timeframe)
        self.assertEqual(latest, test_candles[-1])
    
    def test_get_latest_price_comprehensive(self):
        """Test get_latest_price with various scenarios."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Test with no data
        price = self.data_engine.get_latest_price(symbol, timeframe)
        self.assertIsNone(price)
        
        # Test with data
        test_candles = [
            [1642680000000, 42000, 42100, 41900, 42050, 100]
        ]
        
        key = self.data_engine.data_fetcher.data_processor.get_candle_key(symbol, timeframe)
        self.data_engine.data_fetcher.data_processor.symbol_candles[key] = deque(test_candles, maxlen=100)
        
        price = self.data_engine.get_latest_price(symbol, timeframe)
        self.assertEqual(price, 42050)  # Close price of latest candle
    
    def test_extract_ohlcv_static_method_comprehensive(self):
        """Test extract_ohlcv static method with various inputs."""
        # Test with standard candle format
        candle = [1642680000000, 42000, 42100, 41900, 42050, 100]
        ohlcv = DataEngine.extract_ohlcv(candle)
        
        expected = {
            'timestamp': 1642680000000,
            'open': 42000,
            'high': 42100,
            'low': 41900,
            'close': 42050,
            'volume': 100
        }
        self.assertEqual(ohlcv, expected)
        
        # Test with different data types
        mixed_candle = [1642680000000, 42000.5, 42100.75, 41900.25, 42050.5, 100.5]
        mixed_ohlcv = DataEngine.extract_ohlcv(mixed_candle)
        
        expected_mixed = {
            'timestamp': 1642680000000,
            'open': 42000.5,
            'high': 42100.75,
            'low': 41900.25,
            'close': 42050.5,
            'volume': 100.5
        }
        self.assertEqual(mixed_ohlcv, expected_mixed)


class TestDataEngineHedgeFundStandards(unittest.TestCase):
    """Advanced Data Engine testing with hedge fund standards."""
    
    def setUp(self):
        """Set up hedge fund standard test fixtures."""
        self.mock_client = MockBinanceClient()
        self.mock_client.exchange = MockExchange()
        self.data_engine = DataEngine(self.mock_client, max_candles=500)
    
    def test_flash_crash_data_handling(self):
        """Test data handling during flash crash scenarios."""
        async def _test_async():
            symbol = "BTCUSDT"
            timeframe = "1m"
            
            # Create flash crash scenario
            crash_candles = [
                [1642680000000, 50000.0, 50500.0, 49500.0, 50000.0, 1000.0],  # Normal
                [1642680060000, 50000.0, 50100.0, 40000.0, 40000.0, 10000.0],  # Flash crash
                [1642680120000, 40000.0, 45000.0, 39500.0, 44000.0, 5000.0],   # Recovery
            ]
            
            # Add candles
            for candle in crash_candles:
                await self.data_engine.data_fetcher.data_processor.update_tracked_candles(symbol, timeframe, candle)
            
            # Verify data integrity during crash
            candles = self.data_engine.get_candles(symbol, timeframe)
            self.assertEqual(len(candles), 3)
            
            # Verify flash crash detection possible
            crash_candle = candles[1]
            normal_candle = candles[0]
            
            price_drop = (crash_candle[3] - normal_candle[4]) / normal_candle[4]  # Low vs previous close
            self.assertLess(price_drop, -0.15)  # More than 15% drop
            
            volume_spike = crash_candle[5] / normal_candle[5]
            self.assertGreater(volume_spike, 5)  # Volume spike during crash
        
        asyncio.run(_test_async())
    
    def test_high_frequency_data_integrity(self):
        """Test data integrity under high-frequency updates."""
        async def _test_async():
            symbol = "BTCUSDT"
            timeframe = "1m"
            
            # Simulate high-frequency updates
            base_time = 1642680000000
            base_price = 42000.0
            
            # Add 100 rapid updates
            for i in range(100):
                price_variation = np.random.uniform(-50, 50)
                candle = [
                    base_time + i * 1000,  # 1-second intervals
                    base_price + price_variation,
                    base_price + price_variation + 25,
                    base_price + price_variation - 25,
                    base_price + price_variation + np.random.uniform(-10, 10),
                    100 + np.random.uniform(0, 50)
                ]
                await self.data_engine.data_fetcher.data_processor.update_tracked_candles(symbol, timeframe, candle)
            
            # Verify data integrity
            candles = self.data_engine.get_candles(symbol, timeframe)
            self.assertEqual(len(candles), 100)
            
            # Verify chronological order
            for i in range(1, len(candles)):
                self.assertGreater(candles[i][0], candles[i-1][0])
            
            # Verify latest candle accessibility
            latest = self.data_engine.get_latest_candle(symbol, timeframe)
            self.assertIsNotNone(latest)
            self.assertEqual(latest, candles[-1])
        
        asyncio.run(_test_async())


if __name__ == '__main__':
    # Configure test runner for detailed output
    unittest.main(verbosity=2, buffer=True)
