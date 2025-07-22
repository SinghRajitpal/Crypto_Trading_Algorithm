"""
Unit tests for DataProcessor module.

This test suite covers:
1. Circular buffer implementation
2. Symbol-timeframe key management
3. Candle storage and retrieval
4. Memory limit enforcement
5. Data structure integrity
"""

import unittest
from unittest.mock import patch
import sys
import os
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor(max_candles=3)
        self.sample_candles = [
            [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
            [1642680120000, 42150.0, 42300.0, 42100.0, 42250.0, 110.0],
            [1642680180000, 42250.0, 42400.0, 42200.0, 42350.0, 115.0],
        ]
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor(max_candles=100)
        
        self.assertEqual(processor.max_candles, 100)
        self.assertIsInstance(processor.symbol_candles, dict)
        self.assertEqual(len(processor.symbol_candles), 0)
    
    def test_initialization_edge_cases(self):
        """Test DataProcessor initialization with edge cases."""
        # Test with 0 max_candles
        processor_zero = DataProcessor(max_candles=0)
        self.assertEqual(processor_zero.max_candles, 0)
        
        # Test with 1 max_candles
        processor_one = DataProcessor(max_candles=1)
        self.assertEqual(processor_one.max_candles, 1)
        
        # Test with very large max_candles
        processor_large = DataProcessor(max_candles=1000000)
        self.assertEqual(processor_large.max_candles, 1000000)
    
    def test_get_candle_key(self):
        """Test candle key generation."""
        # Test normal cases
        key1 = self.processor.get_candle_key("BTCUSDT", "1m")
        self.assertEqual(key1, "BTCUSDT_1m")
        
        key2 = self.processor.get_candle_key("ETHUSDT", "5m")
        self.assertEqual(key2, "ETHUSDT_5m")
        
        key3 = self.processor.get_candle_key("BTC/USDT", "1h")
        self.assertEqual(key3, "BTC/USDT_1h")
    
    def test_get_candle_key_edge_cases(self):
        """Test candle key generation with edge cases."""
        # Test with empty strings
        key_empty_symbol = self.processor.get_candle_key("", "1m")
        self.assertEqual(key_empty_symbol, "_1m")
        
        key_empty_tf = self.processor.get_candle_key("BTCUSDT", "")
        self.assertEqual(key_empty_tf, "BTCUSDT_")
        
        # Test with special characters
        key_special = self.processor.get_candle_key("BTC-USDT", "1m")
        self.assertEqual(key_special, "BTC-USDT_1m")
    
    async def test_update_tracked_candles_basic(self):
        """Test basic candle tracking functionality."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Add first candle
        await self.processor.update_tracked_candles(symbol, timeframe, self.sample_candles[0])
        
        candles = self.processor.get_candles(symbol, timeframe)
        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0], self.sample_candles[0])
        
        # Add second candle
        await self.processor.update_tracked_candles(symbol, timeframe, self.sample_candles[1])
        
        candles = self.processor.get_candles(symbol, timeframe)
        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[1], self.sample_candles[1])
    
    async def test_update_tracked_candles_max_limit(self):
        """Test that max_candles limit is enforced."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Add all sample candles (4 candles, but max is 3)
        for candle in self.sample_candles:
            await self.processor.update_tracked_candles(symbol, timeframe, candle)
        
        candles = self.processor.get_candles(symbol, timeframe)
        
        # Should only keep the last 3 candles
        self.assertEqual(len(candles), 3)
        
        # Should be the last 3 candles from sample_candles
        expected_candles = self.sample_candles[-3:]
        self.assertEqual(candles, expected_candles)
    
    async def test_multiple_symbols_and_timeframes(self):
        """Test handling multiple symbol-timeframe pairs."""
        # Add candles for different symbols and timeframes
        await self.processor.update_tracked_candles("BTCUSDT", "1m", self.sample_candles[0])
        await self.processor.update_tracked_candles("BTCUSDT", "5m", self.sample_candles[1])
        await self.processor.update_tracked_candles("ETHUSDT", "1m", self.sample_candles[2])
        
        # Verify each pair is stored separately
        btc_1m = self.processor.get_candles("BTCUSDT", "1m")
        btc_5m = self.processor.get_candles("BTCUSDT", "5m")
        eth_1m = self.processor.get_candles("ETHUSDT", "1m")
        
        self.assertEqual(len(btc_1m), 1)
        self.assertEqual(len(btc_5m), 1)
        self.assertEqual(len(eth_1m), 1)
        
        self.assertEqual(btc_1m[0], self.sample_candles[0])
        self.assertEqual(btc_5m[0], self.sample_candles[1])
        self.assertEqual(eth_1m[0], self.sample_candles[2])
    
    def test_get_candles_empty(self):
        """Test get_candles for non-existent symbol-timeframe pairs."""
        # Test non-existent pair
        candles = self.processor.get_candles("NONEXISTENT", "1m")
        self.assertEqual(candles, [])
        
        # Test empty processor
        empty_processor = DataProcessor(max_candles=100)
        candles_empty = empty_processor.get_candles("BTCUSDT", "1m")
        self.assertEqual(candles_empty, [])
    
    async def test_get_latest_candle(self):
        """Test get_latest_candle functionality."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Test with no data
        latest = self.processor.get_latest_candle(symbol, timeframe)
        self.assertIsNone(latest)
        
        # Add candles
        await self.processor.update_tracked_candles(symbol, timeframe, self.sample_candles[0])
        await self.processor.update_tracked_candles(symbol, timeframe, self.sample_candles[1])
        
        # Test latest candle
        latest = self.processor.get_latest_candle(symbol, timeframe)
        self.assertEqual(latest, self.sample_candles[1])
    
    def test_get_all_symbols(self):
        """Test get_all_symbols functionality."""
        # Test empty processor
        symbols = self.processor.get_all_symbols()
        self.assertEqual(symbols, [])
        
        # Add some data
        async def add_test_data():
            await self.processor.update_tracked_candles("BTCUSDT", "1m", self.sample_candles[0])
            await self.processor.update_tracked_candles("BTCUSDT", "5m", self.sample_candles[1])
            await self.processor.update_tracked_candles("ETHUSDT", "1m", self.sample_candles[2])
        
        import asyncio
        asyncio.run(add_test_data())
        
        symbols = self.processor.get_all_symbols()
        expected_symbols = ["BTCUSDT_1m", "BTCUSDT_5m", "ETHUSDT_1m"]
        
        self.assertEqual(len(symbols), 3)
        for symbol in expected_symbols:
            self.assertIn(symbol, symbols)
    
    async def test_circular_buffer_behavior(self):
        """Test that the circular buffer behaves correctly."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Add exactly max_candles
        for i in range(self.processor.max_candles):
            await self.processor.update_tracked_candles(symbol, timeframe, self.sample_candles[i % len(self.sample_candles)])
        
        candles = self.processor.get_candles(symbol, timeframe)
        self.assertEqual(len(candles), self.processor.max_candles)
        
        # Add one more candle - should replace the oldest
        new_candle = [1642680240000, 42350.0, 42450.0, 42300.0, 42400.0, 120.0]
        await self.processor.update_tracked_candles(symbol, timeframe, new_candle)
        
        candles_after = self.processor.get_candles(symbol, timeframe)
        
        # Should still have max_candles
        self.assertEqual(len(candles_after), self.processor.max_candles)
        
        # Latest candle should be the new one
        self.assertEqual(candles_after[-1], new_candle)
        
        # Should not contain the first candle anymore
        self.assertNotEqual(candles_after[0], candles[0])
    
    def test_data_structure_integrity(self):
        """Test that internal data structures maintain integrity."""
        # Verify internal structure
        self.assertIsInstance(self.processor.symbol_candles, dict)
        
        async def test_internal_structure():
            symbol = "BTCUSDT"
            timeframe = "1m"
            
            await self.processor.update_tracked_candles(symbol, timeframe, self.sample_candles[0])
            
            key = self.processor.get_candle_key(symbol, timeframe)
            
            # Verify internal deque is created
            self.assertIn(key, self.processor.symbol_candles)
            self.assertIsInstance(self.processor.symbol_candles[key], deque)
            
            # Verify maxlen is set correctly
            self.assertEqual(self.processor.symbol_candles[key].maxlen, self.processor.max_candles)
        
        import asyncio
        asyncio.run(test_internal_structure())
    
    async def test_concurrent_access_simulation(self):
        """Test behavior under simulated concurrent access."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Simulate rapid updates
        for candle in self.sample_candles:
            await self.processor.update_tracked_candles(symbol, timeframe, candle)
            
            # Interleave with get operations
            current_candles = self.processor.get_candles(symbol, timeframe)
            latest = self.processor.get_latest_candle(symbol, timeframe)
            
            # Verify consistency
            if current_candles:
                self.assertEqual(latest, current_candles[-1])
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Create processor with larger limit
        large_processor = DataProcessor(max_candles=1000)
        
        # Verify it doesn't pre-allocate memory
        self.assertEqual(len(large_processor.symbol_candles), 0)
        
        # Add data and verify structure
        async def test_memory():
            for i in range(10):
                candle = [1642680000000 + i*60000, 42000.0 + i, 42100.0 + i, 41900.0 + i, 42050.0 + i, 100.0 + i]
                await large_processor.update_tracked_candles("BTCUSDT", "1m", candle)
            
            # Should only have one key
            self.assertEqual(len(large_processor.symbol_candles), 1)
            
            # Should have exactly 10 candles
            candles = large_processor.get_candles("BTCUSDT", "1m")
            self.assertEqual(len(candles), 10)
        
        import asyncio
        asyncio.run(test_memory())
    
    def test_edge_case_with_zero_max_candles(self):
        """Test edge case with zero max_candles."""
        zero_processor = DataProcessor(max_candles=0)
        
        async def test_zero_max():
            # Should handle gracefully
            await zero_processor.update_tracked_candles("BTCUSDT", "1m", self.sample_candles[0])
            
            # Should return empty
            candles = zero_processor.get_candles("BTCUSDT", "1m")
            self.assertEqual(len(candles), 0)
            
            latest = zero_processor.get_latest_candle("BTCUSDT", "1m")
            self.assertIsNone(latest)
        
        import asyncio
        asyncio.run(test_zero_max())
    
    def test_candle_data_types(self):
        """Test handling of different candle data types."""
        async def test_data_types():
            # Test with integer values
            int_candle = [1642680000000, 42000, 42100, 41900, 42050, 100]
            await self.processor.update_tracked_candles("BTCUSDT", "1m", int_candle)
            
            # Test with float values
            float_candle = [1642680060000, 42000.5, 42100.75, 41900.25, 42050.5, 100.5]
            await self.processor.update_tracked_candles("BTCUSDT", "1m", float_candle)
            
            candles = self.processor.get_candles("BTCUSDT", "1m")
            self.assertEqual(len(candles), 2)
            
            # Verify both types are stored correctly
            self.assertEqual(candles[0], int_candle)
            self.assertEqual(candles[1], float_candle)
        
        import asyncio
        asyncio.run(test_data_types())
    
    def test_timestamp_ordering_assumption(self):
        """Test assumptions about timestamp ordering."""
        async def test_ordering():
            # Add candles in random order
            candles = [
                [1642680120000, 42150.0, 42300.0, 42100.0, 42250.0, 110.0],  # Third by time
                [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],  # First by time
                [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],  # Second by time
            ]
            
            for candle in candles:
                await self.processor.update_tracked_candles("BTCUSDT", "1m", candle)
            
            retrieved_candles = self.processor.get_candles("BTCUSDT", "1m")
            
            # Should preserve insertion order, not timestamp order
            self.assertEqual(retrieved_candles[0][0], 1642680120000)  # First inserted
            self.assertEqual(retrieved_candles[1][0], 1642680000000)  # Second inserted
            self.assertEqual(retrieved_candles[2][0], 1642680060000)  # Third inserted
        
        import asyncio
        asyncio.run(test_ordering())


if __name__ == '__main__':
    unittest.main(verbosity=2)
