import asyncio
import time
import statistics
from datetime import datetime
from data.data_engine import DataEngine
from algorithm.algo_engine import AlgoEngine
from algorithm.strategies.base_strategy import BaseStrategy
from binance_exchange import BinanceClient
import config

class BenchmarkStrategy(BaseStrategy):
    """Strategy used only for benchmarking data flow."""
    def __init__(self):
        super().__init__({}, "benchmark_strategy")
        self.processing_times = []
    
    def get_required_indicators(self) -> list:
        """Return an empty list as this benchmark strategy doesn't need any indicators."""
        return []
        
    async def _generate_signals(self, data, indicator_data, symbol):
        # Just measure data access times, don't generate actual signals
        return None

async def benchmark_data_flow():
    print(f"Starting data flow benchmark at {datetime.now().strftime('%H:%M:%S')}")
    
    # Initialize components
    client = BinanceClient(testnet=True)
    data_engine = DataEngine(binance_client=client, max_candles=100)
    algo_engine = AlgoEngine(data_engine=data_engine)
    benchmark_strategy = BenchmarkStrategy()
    
    # Start data collection
    data_task = asyncio.create_task(data_engine.run())
    
    # Wait for initial data collection
    print("Waiting for initial data collection...")
    await asyncio.sleep(10)
    
    # Benchmark symbol
    symbol, timeframe = config.symbols[0]  # Use first configured symbol
    
    # Benchmark 1: Basic data retrieval (already in your script)
    num_runs = 100
    retrieval_times = []
    
    print(f"\nBenchmarking data retrieval for {symbol} ({timeframe})...")
    for i in range(num_runs):
        start_time = time.time()
        
        # Get candles
        candles = data_engine.get_candles(symbol, timeframe)
        
        # Get latest candle
        latest = data_engine.get_latest_candle(symbol, timeframe)
        
        # Get latest price
        price = data_engine.get_latest_price(symbol, timeframe)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        retrieval_times.append(elapsed_ms)
        
        if i % 10 == 0:
            print(f"Run {i}: Retrieved {len(candles)} candles in {elapsed_ms:.2f}ms")
        
        # Small delay between runs
        await asyncio.sleep(0.01)
    
    print_statistics("Data Retrieval Times", retrieval_times)
    
    # Benchmark 2: End-to-end data flow (websocket → algo_engine.process_signals)
    print(f"\nBenchmarking end-to-end data flow (websocket → algo_engine)...")
    flow_times = []
    
    for i in range(50):
        start_time = time.time()
        
        # This calls all the way through data_engine to get_candles and then processes those candles
        signal = await algo_engine.process_signals(symbol, timeframe, benchmark_strategy)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        flow_times.append(elapsed_ms)
        
        print(f"Run {i}: End-to-end processing in {elapsed_ms:.2f}ms")
        await asyncio.sleep(0.05)  # Allow time for new data
    
    print_statistics("End-to-End Flow Times", flow_times)
    
    # Benchmark 3: Calculate indicators (already in your script)
    if hasattr(data_engine, "calculate_atr_volatility"):
        print("\nBenchmarking indicator calculation...")
        indicator_times = []
        
        for i in range(10):
            start_time = time.time()
            
            # Calculate ATR
            atr = data_engine.calculate_atr_volatility(symbol, period=14)
            
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            indicator_times.append(elapsed_ms)
            
            print(f"Run {i}: Calculated ATR ({atr:.6f}) in {elapsed_ms:.2f}ms")
        
        print_statistics("Indicator Calculation Times", indicator_times)
    
    # Stop data collection
    data_task.cancel()
    try:
        await data_task
    except asyncio.CancelledError:
        pass
    
    # Properly close the client connection
    if hasattr(client, 'close') and callable(client.close):
        try:
            await client.close()
        except Exception as e:
            print(f"Error while closing client: {e}")
    
    print(f"\nBenchmark completed at {datetime.now().strftime('%H:%M:%S')}")

def print_statistics(title, times):
    """Print statistics for a set of benchmark times."""
    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"\n{title}:")
    print(f"Average: {avg_time:.2f}ms")
    print(f"Median: {median_time:.2f}ms")
    print(f"Min: {min_time:.2f}ms")
    print(f"Max: {max_time:.2f}ms")
    print(f"StdDev: {stdev:.2f}ms")
    print(f"Throughput: {1000/avg_time:.1f} operations/second")

if __name__ == "__main__":
    try:
        asyncio.run(benchmark_data_flow())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")