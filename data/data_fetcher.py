import ccxt.pro as ccxt
import asyncio
import time
from datetime import datetime
import sys
import os

# Add parent directory to path to find config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Use the symbols from config
symbol_timeframes = config.symbols

client = ccxt.binanceusdm()

async def watch_ohlcv(symbol, timeframe):
    last_printed = None
    
    while True:
        try:
            candles = await client.watch_ohlcv(symbol, timeframe)
            now = time.time() * 1000
            latest = candles[-1]
            
            # Check if this is a closed candle
            if latest[0] != last_printed and now - latest[0] > client.parse_timeframe(timeframe) * 1000:
                # Convert millisecond timestamp to readable date format
                readable_time = datetime.fromtimestamp(latest[0]/1000).strftime('%H:%M:%S %d/%m/%Y')
                print(f"{symbol} ({timeframe}) | Time: {readable_time} | Open: {latest[1]} | High: {latest[2]} | Low: {latest[3]} | Close: {latest[4]} | Volume: {latest[5]}")
                last_printed = latest[0]
                
        except Exception as e:
            print(f"Error: {symbol}/{timeframe} - {e}")
            await asyncio.sleep(1)

async def main():
    tasks = []
    
    # Create a task for each symbol-timeframe pair
    for symbol, timeframe in symbol_timeframes:
        tasks.append(watch_ohlcv(symbol, timeframe))
    
    try:
        await asyncio.gather(*tasks)
    finally:
        await client.close()

asyncio.run(main())

