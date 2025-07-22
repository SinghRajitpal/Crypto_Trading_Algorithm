"""
Integration tests for Data Engine and Algorithm Engine interaction.

This test suite covers:
1. End-to-end data flow from DataEngine to AlgoEngine
2. Signal generation pipeline integration
3. Data consistency and timing synchronization
4. Error propagation and recovery
5. Performance and memory usage patterns
6. Concurrent processing scenarios
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
import time
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.data_engine import DataEngine
from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from algorithm.strategies.base_strategy import BaseStrategy


class MockBinanceClient:
    """Mock Binance client for integration testing."""
    
    def __init__(self, testnet=True):
        self.testnet = testnet
        
        # Create mock exchange with needed methods
        self.exchange = Mock()
        self.exchange.fetch_ohlcv = AsyncMock(return_value=[])
        self.exchange.watch_ohlcv = AsyncMock(return_value=[])
        self.exchange.parse_timeframe = Mock(return_value=60)  # 60 seconds for 1m
        
    async def close(self):
        pass


class IntegrationTestStrategy(BaseStrategy):
    """Test strategy for integration testing."""
    
    def __init__(self):
        super().__init__({"test_param": "value"}, "integration_test_strategy")
        self.signals_generated = []
        
    def get_required_indicators(self):
        # Use simpler indicators that work with small data sets
        return ["sma_3", "sma_5"]
        
    async def calculate_signals(self, data, symbol):
        """Override to ensure timestamp control."""
        # Call the parent implementation
        signal = await super().calculate_signals(data, symbol)
        
        if signal:
            # Force timestamp to None so AlgoEngine can set it
            signal.timestamp = None
            
        return signal
        
    async def _generate_signals(self, data, indicator_data, symbol):
        """Generate test signals based on simple logic."""
        signal = TradeSignal(
            action="hold",
            side="none",
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata={
                "reason": f"Integration test signal for {symbol}",
                "data_points": len(data['close']) if 'close' in data else 0,
                "indicators_available": list(indicator_data.keys())
            },
            signal_confidence=0.6,
            timestamp=None  # Let AlgoEngine set the timestamp
        )
        
        self.signals_generated.append(signal)
        return signal


class MockDataFetcher:
    """Mock data fetcher that simulates realistic data flow."""
    
    def __init__(self, binance_client, max_candles):
        self.binance_client = binance_client
        self.max_candles = max_candles
        self.running = False
        
        # Simulate real candle data - store directly for reliable access
        self.mock_data = {
            "BTCUSDT_1m": [
                [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
                [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
                [1642680120000, 42150.0, 42300.0, 42100.0, 42250.0, 110.0],
                [1642680180000, 42250.0, 42400.0, 42200.0, 42350.0, 115.0],
                [1642680240000, 42350.0, 42500.0, 42300.0, 42450.0, 120.0],
            ],
            "ETHUSDT_1m": [
                [1642680000000, 3000.0, 3050.0, 2950.0, 3025.0, 200.0],
                [1642680060000, 3025.0, 3100.0, 3000.0, 3075.0, 210.0],
                [1642680120000, 3075.0, 3150.0, 3050.0, 3125.0, 220.0],
                [1642680180000, 3125.0, 3200.0, 3100.0, 3175.0, 230.0],
                [1642680240000, 3175.0, 3250.0, 3150.0, 3225.0, 240.0],
            ]
        }
        
        # Mock the data_processor too for consistency
        self.data_processor = Mock()
        self.data_processor.get_candles = lambda symbol, timeframe: self.mock_data.get(f"{symbol}_{timeframe}", [])
        self.data_processor.get_latest_candle = lambda symbol, timeframe: (
            self.mock_data.get(f"{symbol}_{timeframe}", [])[-1] 
            if self.mock_data.get(f"{symbol}_{timeframe}", []) else None
        )
        
    async def run(self):
        """Simulate data fetcher running."""
        self.running = True
        await asyncio.sleep(0.1)  # Simulate data collection time
        
    def get_candles(self, symbol, timeframe):
        """Return mock candle data directly."""
        key = f"{symbol}_{timeframe}"
        return self.mock_data.get(key, [])
    
    def get_latest_candle(self, symbol, timeframe):
        """Return the latest candle directly."""
        candles = self.get_candles(symbol, timeframe)
        if candles:
            return candles[-1]
        return None


class TestDataAlgorithmIntegration(unittest.TestCase):
    """Integration test cases for DataEngine and AlgoEngine."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_client = MockBinanceClient()
        self.test_strategy = IntegrationTestStrategy()
        
    def tearDown(self):
        """Clean up after integration tests."""
        pass
        
    def _create_test_engines_with_mock_data(self):
        """Helper method to create engines with properly mocked data."""
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        
        # Create engines
        data_engine = DataEngine(self.mock_client, max_candles=100)
        
        # Mock the data_engine.data_fetcher.get_candles method directly
        data_engine.data_fetcher.get_candles = Mock(side_effect=lambda symbol, timeframe: mock_fetcher.get_candles(symbol, timeframe))
        
        algo_engine = AlgoEngine(data_engine)
        
        # Set mock algo engine on strategy  
        mock_algo_engine = Mock()
        mock_algo_engine.__class__.__name__ = 'AlgoEngine'
        self.test_strategy.set_algo_engine(mock_algo_engine)
        
        return data_engine, algo_engine
    
    @patch('data.data_fetcher.DataFetcher')
    def test_basic_data_algorithm_integration(self, mock_data_fetcher_class):
        """Test basic integration between DataEngine and AlgoEngine."""        
        async def test_integration():
            data_engine, algo_engine = self._create_test_engines_with_mock_data()
            
            # Test data retrieval
            candles = data_engine.get_candles("BTCUSDT", "1m")
            self.assertEqual(len(candles), 5)
            
            # Test signal processing
            signal = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.symbol, "BTCUSDT")
            self.assertEqual(signal.strategy_id, "integration_test_strategy")
            
        asyncio.run(test_integration())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_signal_generation_pipeline(self, mock_data_fetcher_class):
        """Test complete signal generation pipeline."""
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_pipeline():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            # Process signals for multiple symbols
            symbols = ["BTCUSDT", "ETHUSDT"]
            signals = []
            
            for symbol in symbols:
                signal = await algo_engine.process_signals(symbol, "1m", self.test_strategy)
                if signal:
                    signals.append(signal)
            
            # Verify signals were generated
            self.assertEqual(len(signals), 2)
            
            # Verify signal content
            for i, signal in enumerate(signals):
                self.assertEqual(signal.symbol, symbols[i])
                self.assertIn("data_points", signal.metadata)
                self.assertGreater(signal.metadata["data_points"], 0)
                
        asyncio.run(test_pipeline())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_data_consistency_across_engines(self, mock_data_fetcher_class):
        """Test data consistency between DataEngine and AlgoEngine."""
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_consistency():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            
            # Get data from DataEngine
            candles_from_data_engine = data_engine.get_candles("BTCUSDT", "1m")
            
            # Get data through AlgoEngine
            candles_from_algo_engine = algo_engine.data_engine.get_candles("BTCUSDT", "1m")
            
            # Should be identical
            self.assertEqual(candles_from_data_engine, candles_from_algo_engine)
            
            # Verify data integrity
            self.assertEqual(len(candles_from_data_engine), 5)
            self.assertEqual(candles_from_data_engine[0][4], 42050.0)  # First close
            self.assertEqual(candles_from_data_engine[-1][4], 42450.0)  # Last close
            
        asyncio.run(test_consistency())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_timestamp_synchronization(self, mock_data_fetcher_class):
        """Test timestamp handling and synchronization."""
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_timestamps():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            # Process signal and check timestamp
            start_time = time.time() * 1000  # Milliseconds
            
            signal = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            
            end_time = time.time() * 1000
            
            # Verify signal has timestamp
            self.assertIsNotNone(signal.timestamp)
            
            # The AlgoEngine should set the timestamp to current time, not candle time
            # Allow some tolerance for test execution time
            self.assertGreaterEqual(signal.timestamp, start_time - 1000)  # 1 second tolerance
            self.assertLessEqual(signal.timestamp, end_time + 1000)  # 1 second tolerance
            
            # Verify candle timestamps are in correct format
            candles = data_engine.get_candles("BTCUSDT", "1m")
            for candle in candles:
                timestamp = candle[0]
                # Should be millisecond timestamps (13+ digits)
                self.assertGreaterEqual(len(str(int(timestamp))), 13)
                
        asyncio.run(test_timestamps())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_signal_throttling_integration(self, mock_data_fetcher_class):
        """Test signal throttling behavior in integration."""
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_throttling():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            # Process first signal
            signal1 = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            self.assertIsNotNone(signal1)
            
            # Process second signal immediately (should be throttled)
            signal2 = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            self.assertIsNone(signal2)
            
            # Verify only one signal was generated by strategy
            self.assertEqual(len(self.test_strategy.signals_generated), 1)
            
        asyncio.run(test_throttling())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_error_propagation(self, mock_data_fetcher_class):
        """Test error handling and propagation between engines."""
        # Create mock fetcher that raises error
        mock_fetcher = Mock()
        mock_fetcher.get_candles = Mock(side_effect=Exception("Data fetch error"))
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_errors():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            # Should handle error gracefully
            signal = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            
            # Should return None due to no data
            self.assertIsNone(signal)
            
        asyncio.run(test_errors())
    
    @patch('data.data_fetcher.DataFetcher')
    @patch('data.indicators.Indicators')
    def test_indicator_calculation_integration(self, mock_indicators_class, mock_data_fetcher_class):
        """Test integration with indicator calculation."""
        # Setup mocks
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        mock_data_fetcher_class.return_value = mock_fetcher
        
        mock_indicators = Mock()
        mock_indicators_class.return_value = mock_indicators
        mock_indicators.calculate_indicators = AsyncMock(return_value={
            'sma_3': [42200.0, 42250.0, 42300.0],
            'sma_5': [42100.0, 42150.0, 42200.0, 42250.0, 42300.0]
        })
        
        async def test_indicators():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            signal = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            
            self.assertIsNotNone(signal)
            
            # Verify indicators were available to strategy
            self.assertIn("indicators_available", signal.metadata)
            indicators_list = signal.metadata["indicators_available"]
            self.assertIn("sma_3", indicators_list)
            self.assertIn("sma_5", indicators_list)
            
        asyncio.run(test_indicators())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_multiple_symbol_processing(self, mock_data_fetcher_class):
        """Test processing multiple symbols concurrently."""
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_multiple_symbols():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            symbols = ["BTCUSDT", "ETHUSDT"]
            tasks = []
            
            # Process signals for multiple symbols concurrently
            for symbol in symbols:
                task = algo_engine.process_signals(symbol, "1m", self.test_strategy)
                tasks.append(task)
            
            signals = await asyncio.gather(*tasks)
            
            # Verify all signals were processed
            self.assertEqual(len(signals), 2)
            for signal in signals:
                self.assertIsNotNone(signal)
            
            # Verify symbols are correct
            signal_symbols = [signal.symbol for signal in signals]
            self.assertEqual(set(signal_symbols), set(symbols))
            
        asyncio.run(test_multiple_symbols())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_data_engine_algorithm_engine_lifecycle(self, mock_data_fetcher_class):
        """Test complete lifecycle of DataEngine and AlgoEngine integration."""
        # Create a proper mock fetcher that simulates running behavior
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        
        # Mock the run method to actually set running=True
        original_run = mock_fetcher.run
        async def mock_run():
            mock_fetcher.running = True
            await original_run()
            
        mock_fetcher.run = mock_run
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_lifecycle():
            # 1. Initialize engines
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            # 2. Start data engine
            data_task = asyncio.create_task(data_engine.run())
            await asyncio.sleep(0.1)  # Give it more time to start
            
            self.assertTrue(data_engine.running)
            
            # 3. Process some signals
            signal = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            self.assertIsNotNone(signal)
            
            # 4. Stop data engine
            data_task.cancel()
            try:
                await data_task
            except asyncio.CancelledError:
                pass
                
            self.assertFalse(data_engine.running)
            
            # 5. Stop algo engine
            await algo_engine.stop()
            self.assertFalse(algo_engine.running)
            
        asyncio.run(test_lifecycle())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_memory_usage_patterns(self, mock_data_fetcher_class):
        """Test memory usage patterns in integration."""
        mock_fetcher = MockDataFetcher(self.mock_client, 10)  # Small max_candles
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_memory():
            data_engine = DataEngine(self.mock_client, max_candles=10)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            # Process signals multiple times
            for _ in range(20):  # More than max_candles
                signal = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
                if signal:
                    # Verify signal doesn't hold onto large data
                    self.assertLessEqual(len(signal.metadata), 10)  # Reasonable metadata size
            
            # Verify data engine respects max_candles
            candles = data_engine.get_candles("BTCUSDT", "1m")
            self.assertEqual(len(candles), 5)  # Should be limited by mock data
            
        asyncio.run(test_memory())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_data_freshness_detection(self, mock_data_fetcher_class):
        """Test data freshness detection and change handling."""
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_freshness():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            # Process initial signal
            signal1 = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            self.assertIsNotNone(signal1)
            
            # Mock data hasn't changed, so should be throttled
            signal2 = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            self.assertIsNone(signal2)
            
            # Simulate data change by modifying mock data
            mock_fetcher.mock_data["BTCUSDT_1m"].append(
                [1642680300000, 42450.0, 42550.0, 42400.0, 42500.0, 125.0]
            )
            
            # Should detect change and process new signal
            signal3 = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            self.assertIsNotNone(signal3)
            
        asyncio.run(test_freshness())
    
    @patch('data.data_fetcher.DataFetcher')  
    def test_signal_metadata_enrichment(self, mock_data_fetcher_class):
        """Test signal metadata enrichment through the pipeline."""
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_metadata():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            signal = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            
            self.assertIsNotNone(signal)
            self.assertIn("data_points", signal.metadata)
            self.assertIn("reason", signal.metadata)
            self.assertEqual(signal.metadata["data_points"], 5)  # From mock data
            
            # Verify timestamp was set by algo engine
            self.assertGreater(signal.timestamp, 0)
            
        asyncio.run(test_metadata())
    
    @patch('data.data_fetcher.DataFetcher')
    def test_state_isolation_between_symbols(self, mock_data_fetcher_class):
        """Test state isolation between different symbols."""
        mock_fetcher = MockDataFetcher(self.mock_client, 100)
        mock_data_fetcher_class.return_value = mock_fetcher
        
        async def test_isolation():
            data_engine = DataEngine(self.mock_client, max_candles=100)
            algo_engine = AlgoEngine(data_engine)
            self.test_strategy.set_algo_engine(algo_engine)
            
            # Process signals for different symbols
            btc_signal = await algo_engine.process_signals("BTCUSDT", "1m", self.test_strategy)
            eth_signal = await algo_engine.process_signals("ETHUSDT", "1m", self.test_strategy)
            
            self.assertIsNotNone(btc_signal)
            self.assertIsNotNone(eth_signal)
            
            # Verify signal states are isolated
            btc_key = "BTCUSDT_1m"
            eth_key = "ETHUSDT_1m"
            
            self.assertIn(btc_key, algo_engine._last_signal_states)
            self.assertIn(eth_key, algo_engine._last_signal_states)
            
            btc_state = algo_engine._last_signal_states[btc_key]
            eth_state = algo_engine._last_signal_states[eth_key]
            
            # States should be different (different data hashes)
            self.assertNotEqual(btc_state['data_hash'], eth_state['data_hash'])
            
        asyncio.run(test_isolation())


if __name__ == '__main__':
    unittest.main(verbosity=2)
