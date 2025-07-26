"""
Comprehensive Performance Benchmark Test Suite
Senior Quantitative Trading Systems Testing - Production-Grade Validation

This test suite provides institutional-grade performance benchmarking including:
- Processing speed benchmarks (<30ms per bar requirement)
- Memory usage monitoring and leak detection
- Scalability testing with multiple assets
- Resource utilization profiling
- Concurrent processing validation
- System stress testing under load

Critical Test Vectors:
1. Processing speed compliance with institutional standards
2. Memory efficiency during extended backtests
3. Scalability across increasing asset counts
4. Resource cleanup and garbage collection
5. Performance regression detection
"""

import asyncio
import unittest
import os
import sys
import time
import psutil
import gc
import warnings
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import threading
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtest.backtesting_engine import BacktestingEngine
from algorithm.strategies.ma_crossover import MACrossoverStrategy
from backtest.metrics import Metrics
from backtest.visualizer import QuantStatsVisualizer


class PerformanceBenchmarkFetcher:
    """High-performance mock fetcher for benchmark testing."""
    
    def __init__(self, data_size="medium"):
        self.data_size = data_size
        self.fetch_count = 0
        self.close_called = False
        
        # Pre-generate data to avoid generation overhead during benchmarks
        self._pregenerate_data()
    
    def _pregenerate_data(self):
        """Pre-generate test data for consistent benchmarking."""
        if self.data_size == "small":
            self.num_points = 288    # 1 day of 5m data
        elif self.data_size == "medium":
            self.num_points = 2016   # 1 week of 5m data
        elif self.data_size == "large":
            self.num_points = 8640   # 1 month of 5m data
        else:  # extra_large
            self.num_points = 105120 # 1 year of 5m data
        
        # Generate base price data
        np.random.seed(42)  # Consistent data for benchmarking
        self.price_data = {}
        
        base_prices = {
            "BTCUSDT": 50000,
            "ETHUSDT": 3000,
            "BNBUSDT": 300,
            "ADAUSDT": 1.5,
            "SOLUSDT": 100
        }
        
        for symbol, base_price in base_prices.items():
            prices = []
            current_price = base_price
            
            for i in range(self.num_points):
                # Simple random walk with slight upward bias
                change = np.random.normal(0.0001, 0.01)  # 0.01% change with 1% volatility
                current_price *= (1 + change)
                
                # Generate OHLCV
                open_price = current_price * (1 + np.random.normal(0, 0.001))
                high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.002)))
                low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.002)))
                close_price = current_price
                volume = np.random.uniform(100, 500)
                
                prices.append([open_price, high_price, low_price, close_price, volume])
            
            self.price_data[symbol] = prices
    
    async def download_ohlcv(self, symbol, timeframe, start, end):
        """Return pre-generated OHLCV data."""
        self.fetch_count += 1
        
        # Create date range
        date_range = pd.date_range(start=start, end=end, freq='5min')
        num_points = min(len(date_range), self.num_points)
        
        # Get pre-generated data
        if symbol in self.price_data:
            data = self.price_data[symbol][:num_points]
        else:
            # Fallback for unknown symbols
            data = self.price_data["BTCUSDT"][:num_points]
        
        # Trim date range to match data
        date_range = date_range[:len(data)]
        
        return pd.DataFrame(data, index=date_range,
                          columns=['open', 'high', 'low', 'close', 'volume'])
    
    async def fetch_funding_rate(self, symbol, start, end):
        """Return simple funding rate data."""
        date_range = pd.date_range(start=start, end=end, freq='8h')
        rates = np.full(len(date_range), 0.0001)  # 0.01% funding rate
        return pd.Series(rates, index=date_range)
    
    async def close(self):
        """Mark fetcher as closed."""
        self.close_called = True


class TestProcessingSpeedBenchmarks(unittest.TestCase):
    """Test processing speed benchmarks for institutional compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.institutional_speed_limit = 0.030  # 30ms per bar
        self.symbols = [("BTCUSDT", "5m")]
        self.strategy = MACrossoverStrategy()
        self.start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        self.end_dt = datetime(2024, 1, 8, tzinfo=UTC)  # 1 week
    
    def test_single_asset_processing_speed(self):
        """Test processing speed for single asset backtesting."""
        engine = BacktestingEngine(
            symbols=self.symbols,
            strategy=self.strategy,
            start=self.start_dt,
            end=self.end_dt,
            initial_capital=10000.0
        )
        
        engine.fetcher = PerformanceBenchmarkFetcher(data_size="medium")
        
        async def _test_async():
            # Warm up
            await engine.load_data()
            
            # Benchmark the main processing loop
            start_time = time.perf_counter()
            result = await engine.run()
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            
            # Calculate bars processed
            data_points = engine.fetcher.num_points
            time_per_bar = total_time / data_points if data_points > 0 else 0
            
            print(f"Single Asset Benchmark:")
            print(f"Total Time: {total_time:.4f}s")
            print(f"Data Points: {data_points:,}")
            print(f"Time per Bar: {time_per_bar*1000:.2f}ms")
            print(f"Bars per Second: {1/time_per_bar if time_per_bar > 0 else 0:.0f}")
            print(f"Institutional Limit: {self.institutional_speed_limit*1000:.0f}ms")
            
            # Validate performance
            self.assertLess(time_per_bar, self.institutional_speed_limit,
                          f"Processing speed {time_per_bar*1000:.2f}ms exceeds {self.institutional_speed_limit*1000:.0f}ms limit")
            
            # Verify backtest completed successfully
            self.assertIsInstance(result, dict)
            self.assertIn("trades", result)
        
        asyncio.run(_test_async())
    
    def test_multi_asset_processing_speed(self):
        """Test processing speed for multi-asset backtesting."""
        multi_symbols = [
            ("BTCUSDT", "5m"),
            ("ETHUSDT", "5m"),
            ("BNBUSDT", "5m"),
            ("ADAUSDT", "5m"),
            ("SOLUSDT", "5m")
        ]
        
        engine = BacktestingEngine(
            symbols=multi_symbols,
            strategy=self.strategy,
            start=self.start_dt,
            end=self.end_dt,
            initial_capital=10000.0
        )
        
        engine.fetcher = PerformanceBenchmarkFetcher(data_size="medium")
        
        async def _test_async():
            start_time = time.perf_counter()
            result = await engine.run()
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            
            # Calculate processing metrics
            num_symbols = len(multi_symbols)
            data_points = engine.fetcher.num_points
            total_bars = data_points * num_symbols
            time_per_bar = total_time / total_bars if total_bars > 0 else 0
            
            print(f"Multi-Asset Benchmark:")
            print(f"Number of Symbols: {num_symbols}")
            print(f"Total Time: {total_time:.4f}s")
            print(f"Data Points per Symbol: {data_points:,}")
            print(f"Total Bars Processed: {total_bars:,}")
            print(f"Time per Bar: {time_per_bar*1000:.2f}ms")
            print(f"Institutional Limit: {self.institutional_speed_limit*1000:.0f}ms")
            
            # Multi-asset should still meet institutional standards
            self.assertLess(time_per_bar, self.institutional_speed_limit,
                          f"Multi-asset processing {time_per_bar*1000:.2f}ms exceeds {self.institutional_speed_limit*1000:.0f}ms limit")
            
            self.assertIsInstance(result, dict)
        
        asyncio.run(_test_async())
    
    def test_scalability_across_asset_counts(self):
        """Test scalability as number of assets increases."""
        asset_counts = [1, 3, 5, 10, 20]
        performance_results = []
        
        base_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
                       "XRPUSDT", "DOGEUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT",
                       "AVAXUSDT", "UNIUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT",
                       "VETUSDT", "FILUSDT", "TRXUSDT", "ETCUSDT", "XMRUSDT"]
        
        async def test_asset_count(count):
            symbols = [(f"{base_symbols[i % len(base_symbols)]}", "5m") for i in range(count)]
            
            engine = BacktestingEngine(
                symbols=symbols,
                strategy=self.strategy,
                start=self.start_dt,
                end=datetime(2024, 1, 3, tzinfo=UTC),  # Shorter period for scalability test
                initial_capital=10000.0
            )
            
            engine.fetcher = PerformanceBenchmarkFetcher(data_size="small")
            
            start_time = time.perf_counter()
            result = await engine.run()
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            data_points = engine.fetcher.num_points
            total_bars = data_points * count
            time_per_bar = total_time / total_bars if total_bars > 0 else 0
            
            return {
                'asset_count': count,
                'total_time': total_time,
                'time_per_bar': time_per_bar,
                'bars_per_second': 1/time_per_bar if time_per_bar > 0 else 0
            }
        
        async def _test_async():
            for count in asset_counts:
                result = await test_asset_count(count)
                performance_results.append(result)
            
            print("Scalability Analysis:")
            for result in performance_results:
                print(f"Assets: {result['asset_count']:2d} | "
                      f"Time/Bar: {result['time_per_bar']*1000:5.1f}ms | "
                      f"Bars/sec: {result['bars_per_second']:6.0f}")
            
            # Analyze scalability
            single_asset_time = performance_results[0]['time_per_bar']
            max_asset_time = performance_results[-1]['time_per_bar']
            
            scalability_factor = max_asset_time / single_asset_time if single_asset_time > 0 else 0
            
            print(f"Scalability Factor: {scalability_factor:.2f}x")
            
            # Validate reasonable scalability
            self.assertLess(scalability_factor, 5.0,  # Should not be more than 5x slower
                          f"Poor scalability: {scalability_factor:.2f}x slowdown")
            
            # All configurations should meet institutional standards
            for result in performance_results:
                self.assertLess(result['time_per_bar'], self.institutional_speed_limit)
        
        asyncio.run(_test_async())


class TestMemoryUsageBenchmarks(unittest.TestCase):
    """Test memory usage and leak detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.process = psutil.Process()
        self.memory_limit_mb = 1000  # 1GB limit for backtesting
    
    def test_memory_usage_single_backtest(self):
        """Test memory usage for single backtest."""
        # Force garbage collection before test
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        symbols = [("BTCUSDT", "5m")]
        strategy = MACrossoverStrategy()
        
        engine = BacktestingEngine(
            symbols=symbols,
            strategy=strategy,
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 15, tzinfo=UTC),  # 2 weeks
            initial_capital=10000.0
        )
        
        engine.fetcher = PerformanceBenchmarkFetcher(data_size="large")
        
        async def _test_async():
            result = await engine.run()
            
            # Measure peak memory usage
            peak_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            print(f"Memory Usage Analysis:")
            print(f"Initial Memory: {initial_memory:.1f} MB")
            print(f"Peak Memory: {peak_memory:.1f} MB")
            print(f"Memory Increase: {memory_increase:.1f} MB")
            print(f"Memory Limit: {self.memory_limit_mb} MB")
            
            # Validate memory usage
            self.assertLess(peak_memory, self.memory_limit_mb,
                          f"Memory usage {peak_memory:.1f}MB exceeds {self.memory_limit_mb}MB limit")
            
            # Verify backtest completed
            self.assertIsInstance(result, dict)
        
        asyncio.run(_test_async())
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated backtests."""
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        memory_samples = [initial_memory]
        
        async def run_single_backtest():
            symbols = [("BTCUSDT", "5m")]
            strategy = MACrossoverStrategy()
            
            engine = BacktestingEngine(
                symbols=symbols,
                strategy=strategy,
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 3, tzinfo=UTC),  # Short period
                initial_capital=10000.0
            )
            
            engine.fetcher = PerformanceBenchmarkFetcher(data_size="small")
            result = await engine.run()
            
            # Explicit cleanup
            del engine
            gc.collect()
            
            return result
        
        async def _test_async():
            # Run multiple backtests
            num_iterations = 10
            
            for i in range(num_iterations):
                result = await run_single_backtest()
                
                # Sample memory after each run
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                print(f"Iteration {i+1}: {current_memory:.1f} MB")
            
            # Analyze memory trend
            final_memory = memory_samples[-1]
            memory_growth = final_memory - initial_memory
            avg_growth_per_iteration = memory_growth / num_iterations
            
            print(f"\nMemory Leak Analysis:")
            print(f"Initial Memory: {initial_memory:.1f} MB")
            print(f"Final Memory: {final_memory:.1f} MB")
            print(f"Total Growth: {memory_growth:.1f} MB")
            print(f"Growth per Iteration: {avg_growth_per_iteration:.2f} MB")
            
            # Validate no significant memory leaks
            max_acceptable_growth = 50  # 50MB total growth acceptable
            self.assertLess(memory_growth, max_acceptable_growth,
                          f"Memory leak detected: {memory_growth:.1f}MB growth exceeds {max_acceptable_growth}MB")
            
            # Growth per iteration should be minimal
            max_per_iteration = 5  # 5MB per iteration
            self.assertLess(avg_growth_per_iteration, max_per_iteration,
                          f"Memory leak per iteration: {avg_growth_per_iteration:.2f}MB exceeds {max_per_iteration}MB")
        
        asyncio.run(_test_async())
    
    def test_large_dataset_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Test with year-long dataset
        symbols = [("BTCUSDT", "5m")]
        strategy = MACrossoverStrategy()
        
        engine = BacktestingEngine(
            symbols=symbols,
            strategy=strategy,
            start=datetime(2023, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, tzinfo=UTC),  # Full year
            initial_capital=10000.0
        )
        
        engine.fetcher = PerformanceBenchmarkFetcher(data_size="extra_large")
        
        async def _test_async():
            # Monitor memory during execution
            start_time = time.time()
            result = await engine.run()
            end_time = time.time()
            
            peak_memory = self.process.memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory
            execution_time = end_time - start_time
            
            # Calculate data efficiency metrics
            data_points = engine.fetcher.num_points
            memory_per_datapoint = memory_increase / data_points if data_points > 0 else 0
            
            print(f"Large Dataset Memory Analysis:")
            print(f"Dataset Size: {data_points:,} data points")
            print(f"Execution Time: {execution_time:.1f}s")
            print(f"Initial Memory: {initial_memory:.1f} MB")
            print(f"Peak Memory: {peak_memory:.1f} MB")
            print(f"Memory Increase: {memory_increase:.1f} MB")
            print(f"Memory per Data Point: {memory_per_datapoint*1024:.2f} KB")
            
            # Validate efficient memory usage
            max_memory_increase = 300  # 300MB for year of data
            self.assertLess(memory_increase, max_memory_increase,
                          f"Large dataset memory usage {memory_increase:.1f}MB exceeds {max_memory_increase}MB")
            
            # Memory per data point should be reasonable
            max_memory_per_point = 0.01  # 10KB per data point
            self.assertLess(memory_per_datapoint, max_memory_per_point,
                          f"Memory per data point {memory_per_datapoint*1024:.2f}KB exceeds {max_memory_per_point*1024:.0f}KB")
            
            self.assertIsInstance(result, dict)
        
        asyncio.run(_test_async())


class TestConcurrentProcessingBenchmarks(unittest.TestCase):
    """Test concurrent processing capabilities."""
    
    def test_multiple_strategy_concurrent_execution(self):
        """Test concurrent execution of multiple strategies."""
        strategies = [
            MACrossoverStrategy(params={'fast_ma_period': 5, 'slow_ma_period': 20}),
            MACrossoverStrategy(params={'fast_ma_period': 8, 'slow_ma_period': 21}),
            MACrossoverStrategy(params={'fast_ma_period': 10, 'slow_ma_period': 30}),
        ]
        
        async def run_strategy_backtest(strategy, strategy_id):
            symbols = [("BTCUSDT", "5m")]
            
            engine = BacktestingEngine(
                symbols=symbols,
                strategy=strategy,
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 8, tzinfo=UTC),
                initial_capital=10000.0
            )
            
            engine.fetcher = PerformanceBenchmarkFetcher(data_size="medium")
            
            start_time = time.perf_counter()
            result = await engine.run()
            end_time = time.perf_counter()
            
            return {
                'strategy_id': strategy_id,
                'execution_time': end_time - start_time,
                'final_cash': result['final_cash'],
                'trade_count': result['trade_count']
            }
        
        async def _test_async():
            # Test sequential execution
            print("Sequential Execution:")
            sequential_start = time.perf_counter()
            sequential_results = []
            
            for i, strategy in enumerate(strategies):
                result = await run_strategy_backtest(strategy, f"seq_{i}")
                sequential_results.append(result)
                print(f"  Strategy {i}: {result['execution_time']:.3f}s")
            
            sequential_total = time.perf_counter() - sequential_start
            
            # Test concurrent execution
            print("\nConcurrent Execution:")
            concurrent_start = time.perf_counter()
            
            tasks = []
            for i, strategy in enumerate(strategies):
                task = run_strategy_backtest(strategy, f"con_{i}")
                tasks.append(task)
            
            concurrent_results = await asyncio.gather(*tasks)
            concurrent_total = time.perf_counter() - concurrent_start
            
            for i, result in enumerate(concurrent_results):
                print(f"  Strategy {i}: {result['execution_time']:.3f}s")
            
            # Analyze concurrent performance
            speedup = sequential_total / concurrent_total
            efficiency = speedup / len(strategies)
            
            print(f"\nConcurrency Analysis:")
            print(f"Sequential Total: {sequential_total:.3f}s")
            print(f"Concurrent Total: {concurrent_total:.3f}s")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Efficiency: {efficiency:.2%}")
            
            # Validate concurrent execution benefits
            self.assertGreater(speedup, 0.8,  # Should be at least 0.8x (accounting for overhead)
                             f"Concurrent execution speedup {speedup:.2f}x is insufficient")
            
            # Verify all strategies completed successfully
            self.assertEqual(len(concurrent_results), len(strategies))
            for result in concurrent_results:
                self.assertIsInstance(result['final_cash'], (int, float))
        
        asyncio.run(_test_async())


class TestResourceUtilizationBenchmarks(unittest.TestCase):
    """Test system resource utilization."""
    
    def test_cpu_utilization_monitoring(self):
        """Test CPU utilization during backtesting."""
        # Get initial CPU usage
        initial_cpu = psutil.cpu_percent(interval=1)
        
        symbols = [("BTCUSDT", "5m"), ("ETHUSDT", "5m")]
        strategy = MACrossoverStrategy()
        
        engine = BacktestingEngine(
            symbols=symbols,
            strategy=strategy,
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 15, tzinfo=UTC),
            initial_capital=10000.0
        )
        
        engine.fetcher = PerformanceBenchmarkFetcher(data_size="large")
        
        cpu_samples = []
        
        def monitor_cpu():
            """Monitor CPU usage during execution."""
            while True:
                cpu_usage = psutil.cpu_percent(interval=0.5)
                cpu_samples.append(cpu_usage)
                if len(cpu_samples) > 100:  # Limit samples
                    break
        
        async def _test_async():
            # Start CPU monitoring in background
            monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
            monitor_thread.start()
            
            # Run backtest
            start_time = time.perf_counter()
            result = await engine.run()
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            
            # Analyze CPU usage
            if cpu_samples:
                avg_cpu = np.mean(cpu_samples)
                max_cpu = np.max(cpu_samples)
                min_cpu = np.min(cpu_samples)
                
                print(f"CPU Utilization Analysis:")
                print(f"Execution Time: {execution_time:.2f}s")
                print(f"Initial CPU: {initial_cpu:.1f}%")
                print(f"Average CPU: {avg_cpu:.1f}%")
                print(f"Peak CPU: {max_cpu:.1f}%")
                print(f"Min CPU: {min_cpu:.1f}%")
                
                # Validate reasonable CPU usage
                self.assertLess(avg_cpu, 80,  # Should not consistently max out CPU
                              f"Average CPU usage {avg_cpu:.1f}% is too high")
                
                self.assertGreater(avg_cpu, initial_cpu + 5,  # Should show some CPU activity
                                 f"CPU usage {avg_cpu:.1f}% shows insufficient activity")
            
            self.assertIsInstance(result, dict)
        
        asyncio.run(_test_async())
    
    def test_disk_io_efficiency(self):
        """Test disk I/O efficiency during backtesting."""
        # Monitor disk I/O
        initial_io = psutil.disk_io_counters()
        
        symbols = [("BTCUSDT", "5m")]
        strategy = MACrossoverStrategy()
        
        engine = BacktestingEngine(
            symbols=symbols,
            strategy=strategy,
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 8, tzinfo=UTC),
            initial_capital=10000.0
        )
        
        engine.fetcher = PerformanceBenchmarkFetcher(data_size="medium")
        
        async def _test_async():
            result = await engine.run()
            
            # Measure I/O after execution
            final_io = psutil.disk_io_counters()
            
            if initial_io and final_io:
                read_bytes = final_io.read_bytes - initial_io.read_bytes
                write_bytes = final_io.write_bytes - initial_io.write_bytes
                
                print(f"Disk I/O Analysis:")
                print(f"Bytes Read: {read_bytes / 1024 / 1024:.2f} MB")
                print(f"Bytes Written: {write_bytes / 1024 / 1024:.2f} MB")
                print(f"Total I/O: {(read_bytes + write_bytes) / 1024 / 1024:.2f} MB")
                
                # Validate reasonable I/O usage
                max_io_mb = 100  # 100MB max I/O for medium dataset
                total_io_mb = (read_bytes + write_bytes) / 1024 / 1024
                
                self.assertLess(total_io_mb, max_io_mb,
                              f"Disk I/O {total_io_mb:.2f}MB exceeds {max_io_mb}MB limit")
            
            self.assertIsInstance(result, dict)
        
        asyncio.run(_test_async())


class TestPerformanceRegressionDetection(unittest.TestCase):
    """Test performance regression detection."""
    
    def test_performance_baseline_validation(self):
        """Test performance against established baselines."""
        # Established performance baselines (institutional standards)
        baselines = {
            'time_per_bar_ms': 25.0,      # 25ms per bar maximum
            'memory_per_datapoint_kb': 5.0, # 5KB per data point maximum
            'bars_per_second': 40.0,       # 40 bars per second minimum
        }
        
        symbols = [("BTCUSDT", "5m")]
        strategy = MACrossoverStrategy()
        
        engine = BacktestingEngine(
            symbols=symbols,
            strategy=strategy,
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 8, tzinfo=UTC),
            initial_capital=10000.0
        )
        
        engine.fetcher = PerformanceBenchmarkFetcher(data_size="medium")
        
        async def _test_async():
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            result = await engine.run()
            end_time = time.perf_counter()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate performance metrics
            execution_time = end_time - start_time
            data_points = engine.fetcher.num_points
            memory_increase = final_memory - initial_memory
            
            time_per_bar_ms = (execution_time / data_points * 1000) if data_points > 0 else 0
            memory_per_datapoint_kb = (memory_increase * 1024 / data_points) if data_points > 0 else 0
            bars_per_second = (data_points / execution_time) if execution_time > 0 else 0
            
            current_performance = {
                'time_per_bar_ms': time_per_bar_ms,
                'memory_per_datapoint_kb': memory_per_datapoint_kb,
                'bars_per_second': bars_per_second,
            }
            
            print("Performance Baseline Validation:")
            for metric, current_value in current_performance.items():
                baseline_value = baselines[metric]
                
                if 'per_second' in metric:
                    # Higher is better
                    performance_ratio = current_value / baseline_value
                    status = "✓ PASS" if current_value >= baseline_value else "✗ FAIL"
                else:
                    # Lower is better
                    performance_ratio = baseline_value / current_value if current_value > 0 else 0
                    status = "✓ PASS" if current_value <= baseline_value else "✗ FAIL"
                
                print(f"  {metric}: {current_value:.2f} (baseline: {baseline_value:.2f}) "
                      f"[{performance_ratio:.2f}x] {status}")
                
                # Validate against baseline
                if 'per_second' in metric:
                    self.assertGreaterEqual(current_value, baseline_value,
                                          f"{metric} {current_value:.2f} below baseline {baseline_value:.2f}")
                else:
                    self.assertLessEqual(current_value, baseline_value,
                                       f"{metric} {current_value:.2f} exceeds baseline {baseline_value:.2f}")
            
            self.assertIsInstance(result, dict)
        
        asyncio.run(_test_async())


if __name__ == '__main__':
    # Configure test execution
    unittest.main(verbosity=2, buffer=True)
