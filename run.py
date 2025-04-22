#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.insert(0, project_root)

# Now import and run the main script
from main import TradingAlgorithm
from algorithm.strategies.ma_crossover import MACrossoverStrategy
import asyncio

def main():
    try:
        # Create strategy with custom parameters
        strategy = MACrossoverStrategy(params={
            'fast_ma_period': 9,
            'slow_ma_period': 21
        })
        
        # Create and run the trading algorithm with the strategy
        algorithm = TradingAlgorithm(strategy=strategy, testnet=True)
        
        asyncio.run(algorithm.start())
    except KeyboardInterrupt:
        print("\nTrading algorithm stopped by user.")
    except Exception as e:
        print(f"Error running trading algorithm: {e}")

if __name__ == "__main__":
    main() 