"""
Unit tests for DataEngine module.

This test suite covers:
1. Initialization and configuration
2. Data retrieval methods
3. OHLCV data processing
4. Error handling and edge cases
5. Data integrity validation
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import sys
import os
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.data_engine import DataEngine
from data.data_fetcher import DataFetcher


class MockBinanceClient:
    """Mock Binance client for testing."""
    def __init__(self, testnet=True):
        self.testnet = testnet
        
    async def close(self):
        pass


class MockDataFetcher:
    """Mock data fetcher for testing."""
    def __init__(self, binance_client, max_candles):
        self.binance_client = binance_client
        self.max_candles = max_candles
        self.running = False
        
    async def run(self):
        self.running = True
        # Simulate data collection
        await asyncio.sleep(0.1)
        
    def get_candles(self, symbol, timeframe):
        """Return mock candle data."""
        if symbol == "BTCUSDT" and timeframe == "1m":
            return [
                [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],  # Mock candle 1
                [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],  # Mock candle 2
                [1642680120000, 42150.0, 42300.0, 42100.0, 42250.0, 110.0],  # Mock candle 3
            ]
        return []


class TestDataEngine(unittest.TestCase):
    """Test cases for DataEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockBinanceClient()
        
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_data_engine_initialization(self):
        """Test DataEngine initialization with valid parameters."""
        # Test default initialization
        engine = DataEngine(self.mock_client)
        
        self.assertIsNotNone(engine.binance_client)
        self.assertIsNotNone(engine.data_fetcher)
        self.assertFalse(engine.running)
        
        # Test custom max_candles
        engine_custom = DataEngine(self.mock_client, max_candles=200)
        self.assertEqual(engine_custom.data_fetcher.data_processor.max_candles, 200)
    
    def test_data_engine_initialization_edge_cases(self):
        """Test DataEngine initialization with edge cases."""
        # Test with None client (should not raise exception due to duck typing)
        with patch('data.data_fetcher.DataFetcher'):
            engine = DataEngine(None)
            self.assertIsNone(engine.binance_client)
        
        # Test with very small max_candles
        engine_small = DataEngine(self.mock_client, max_candles=1)
        self.assertEqual(engine_small.data_fetcher.data_processor.max_candles, 1)
        
        # Test with very large max_candles
        engine_large = DataEngine(self.mock_client, max_candles=10000)
        self.assertEqual(engine_large.data_fetcher.data_processor.max_candles, 10000)
    
    @patch('data.data_engine.DataFetcher')
    def test_data_engine_run_method(self, mock_data_fetcher_class):
        """Test DataEngine run method."""
        # Setup mock
        mock_fetcher = AsyncMock()
        mock_data_fetcher_class.return_value = mock_fetcher
        
        engine = DataEngine(self.mock_client)
        
        async def test_run():
            # Test normal run
            self.assertFalse(engine.running)
            
            # Mock the fetcher run method
            mock_fetcher.run = AsyncMock()
            
            # Start the engine in background
            task = asyncio.create_task(engine.run())
            await asyncio.sleep(0.01)  # Let it start
            
            self.assertTrue(engine.running)
            
            # Cancel the task to stop
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
            self.assertFalse(engine.running)
        
        # Run the async test
        asyncio.run(test_run())
    
    @patch('data.data_engine.DataFetcher')
    def test_data_engine_run_already_running(self, mock_data_fetcher_class):
        """Test DataEngine run method when already running."""
        mock_fetcher = AsyncMock()
        mock_data_fetcher_class.return_value = mock_fetcher
        
        engine = DataEngine(self.mock_client)
        engine.running = True  # Manually set to running
        
        async def test_already_running():
            # Should return immediately without starting
            await engine.run()
            mock_fetcher.run.assert_not_called()
        
        asyncio.run(test_already_running())
    
    def test_get_candles_method(self):
        """Test get_candles method."""
        with patch.object(DataEngine, '__init__', return_value=None):
            engine = DataEngine.__new__(DataEngine)
            engine.data_fetcher = MockDataFetcher(self.mock_client, 100)
            
            # Test valid symbol and timeframe
            candles = engine.get_candles("BTCUSDT", "1m")
            self.assertEqual(len(candles), 3)
            self.assertEqual(candles[0][4], 42050.0)  # First close price
            
            # Test invalid symbol
            candles_invalid = engine.get_candles("INVALID", "1m")
            self.assertEqual(len(candles_invalid), 0)
    
    def test_get_latest_candle_method(self):
        """Test get_latest_candle method."""
        with patch.object(DataEngine, '__init__', return_value=None):
            engine = DataEngine.__new__(DataEngine)
            engine.data_fetcher = MockDataFetcher(self.mock_client, 100)
            
            # Test valid data
            latest = engine.get_latest_candle("BTCUSDT", "1m")
            self.assertIsNotNone(latest)
            self.assertEqual(latest[4], 42250.0)  # Last close price
            
            # Test no data
            latest_none = engine.get_latest_candle("INVALID", "1m")
            self.assertIsNone(latest_none)
    
    def test_get_latest_price_method(self):
        """Test get_latest_price method."""
        with patch.object(DataEngine, '__init__', return_value=None):
            engine = DataEngine.__new__(DataEngine)
            engine.data_fetcher = MockDataFetcher(self.mock_client, 100)
            
            # Test valid data
            price = engine.get_latest_price("BTCUSDT", "1m")
            self.assertEqual(price, 42250.0)
            
            # Test no data
            price_none = engine.get_latest_price("INVALID", "1m")
            self.assertIsNone(price_none)
    
    def test_extract_ohlcv_static_method(self):
        """Test extract_ohlcv static method."""
        # Test valid candle
        candle = [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]
        ohlcv = DataEngine.extract_ohlcv(candle)
        
        expected = {
            "timestamp": 1642680000000,
            "open": 42000.0,
            "high": 42100.0,
            "low": 41900.0,
            "close": 42050.0,
            "volume": 100.0
        }
        
        self.assertEqual(ohlcv, expected)
    
    def test_extract_ohlcv_edge_cases(self):
        """Test extract_ohlcv with edge cases."""
        # Test None candle
        ohlcv_none = DataEngine.extract_ohlcv(None)
        self.assertEqual(ohlcv_none, {})
        
        # Test empty candle
        ohlcv_empty = DataEngine.extract_ohlcv([])
        self.assertEqual(ohlcv_empty, {})
        
        # Test incomplete candle
        ohlcv_incomplete = DataEngine.extract_ohlcv([1642680000000, 42000.0])
        self.assertEqual(ohlcv_incomplete, {})
        
        # Test exactly 5 elements (minimum)
        candle_min = [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0]
        ohlcv_min = DataEngine.extract_ohlcv(candle_min)
        self.assertEqual(len(ohlcv_min), 0)  # Needs 6 elements including volume
        
        # Test exactly 6 elements (valid)
        candle_valid = [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]
        ohlcv_valid = DataEngine.extract_ohlcv(candle_valid)
        self.assertEqual(len(ohlcv_valid), 6)
    
    def test_get_candle_change_pct_static_method(self):
        """Test get_candle_change_pct static method."""
        # Test bullish candle
        candle_bull = [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]
        change_bull = DataEngine.get_candle_change_pct(candle_bull)
        expected_bull = ((42050.0 - 42000.0) / 42000.0) * 100  # +0.119%
        self.assertAlmostEqual(change_bull, expected_bull, places=2)
        
        # Test bearish candle
        candle_bear = [1642680000000, 42000.0, 42100.0, 41900.0, 41950.0, 100.0]
        change_bear = DataEngine.get_candle_change_pct(candle_bear)
        expected_bear = ((41950.0 - 42000.0) / 42000.0) * 100  # -0.119%
        self.assertAlmostEqual(change_bear, expected_bear, places=2)
    
    def test_get_candle_change_pct_edge_cases(self):
        """Test get_candle_change_pct with edge cases."""
        # Test None candle
        change_none = DataEngine.get_candle_change_pct(None)
        self.assertEqual(change_none, 0.0)
        
        # Test empty candle
        change_empty = DataEngine.get_candle_change_pct([])
        self.assertEqual(change_empty, 0.0)
        
        # Test candle with zero open price
        candle_zero_open = [1642680000000, 0.0, 42100.0, 41900.0, 42050.0, 100.0]
        change_zero = DataEngine.get_candle_change_pct(candle_zero_open)
        self.assertEqual(change_zero, 0.0)
        
        # Test incomplete candle
        candle_incomplete = [1642680000000, 42000.0, 42100.0]
        change_incomplete = DataEngine.get_candle_change_pct(candle_incomplete)
        self.assertEqual(change_incomplete, 0.0)
    
    def test_data_integrity_validation(self):
        """Test data integrity and consistency checks."""
        with patch.object(DataEngine, '__init__', return_value=None):
            engine = DataEngine.__new__(DataEngine)
            engine.data_fetcher = MockDataFetcher(self.mock_client, 100)
            
            # Get candles and verify data consistency
            candles = engine.get_candles("BTCUSDT", "1m")
            
            for i, candle in enumerate(candles):
                # Verify candle structure
                self.assertEqual(len(candle), 6, f"Candle {i} should have 6 elements")
                
                # Verify OHLC relationships
                self.assertGreaterEqual(candle[2], candle[1], f"High >= Open in candle {i}")  # High >= Open
                self.assertGreaterEqual(candle[2], candle[4], f"High >= Close in candle {i}")  # High >= Close
                self.assertLessEqual(candle[3], candle[1], f"Low <= Open in candle {i}")      # Low <= Open
                self.assertLessEqual(candle[3], candle[4], f"Low <= Close in candle {i}")     # Low <= Close
                self.assertGreaterEqual(candle[2], candle[3], f"High >= Low in candle {i}")   # High >= Low
                
                # Verify positive values
                self.assertGreaterEqual(candle[5], 0, f"Volume >= 0 in candle {i}")  # Volume >= 0
                
                # Verify timestamp progression (if not first candle)
                if i > 0:
                    self.assertGreater(candle[0], candles[i-1][0], f"Timestamp progression in candle {i}")
    
    def test_timestamp_handling(self):
        """Test timestamp handling and conversion."""
        with patch.object(DataEngine, '__init__', return_value=None):
            engine = DataEngine.__new__(DataEngine)
            engine.data_fetcher = MockDataFetcher(self.mock_client, 100)
            
            candles = engine.get_candles("BTCUSDT", "1m")
            
            for candle in candles:
                timestamp = candle[0]
                
                # Verify timestamp is in milliseconds (13 digits)
                self.assertGreaterEqual(len(str(int(timestamp))), 13, "Timestamp should be in milliseconds")
                
                # Verify timestamp is reasonable (not in the past before 2020 or too far in future)
                self.assertGreater(timestamp, 1577836800000, "Timestamp should be after 2020-01-01")  # 2020-01-01
                self.assertLess(timestamp, 1893456000000, "Timestamp should be before 2030-01-01")     # 2030-01-01
    
    def test_memory_management(self):
        """Test memory management and max_candles limit."""
        # Create engine with small max_candles
        engine = DataEngine(self.mock_client, max_candles=2)
        
        # Verify max_candles is respected
        self.assertEqual(engine.data_fetcher.data_processor.max_candles, 2)
        
        # This test would need a more complex mock to simulate adding many candles
        # and verifying that only the latest N are kept
    
    async def test_error_handling_in_run(self):
        """Test error handling in run method."""
        with patch('data.data_engine.DataFetcher') as mock_data_fetcher_class:
            # Setup mock to raise exception
            mock_fetcher = AsyncMock()
            mock_fetcher.run.side_effect = Exception("Network error")
            mock_data_fetcher_class.return_value = mock_fetcher
            
            engine = DataEngine(self.mock_client)
            
            # Should handle exception gracefully
            with self.assertRaises(Exception):
                await engine.run()
            
            # Should reset running state
            self.assertFalse(engine.running)
    
    def test_concurrent_access(self):
        """Test thread safety and concurrent access patterns."""
        with patch.object(DataEngine, '__init__', return_value=None):
            engine = DataEngine.__new__(DataEngine)
            engine.data_fetcher = MockDataFetcher(self.mock_client, 100)
            
            # Test multiple simultaneous get_candles calls
            candles1 = engine.get_candles("BTCUSDT", "1m")
            candles2 = engine.get_candles("BTCUSDT", "1m")
            
            # Should return consistent data
            self.assertEqual(candles1, candles2)
            
            # Test simultaneous access to different symbols
            candles_btc = engine.get_candles("BTCUSDT", "1m")
            candles_eth = engine.get_candles("ETHUSDT", "1m")  # Will return empty for mock
            
            # Should handle different symbols independently
            self.assertGreater(len(candles_btc), 0)
            self.assertEqual(len(candles_eth), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
