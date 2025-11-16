import os
import re
import sys

import requests

# Add parent directory to path to find config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def update_config_with_high_liquidity_coins(min_market_cap=200, max_coins=None, timeframe="5m"):
    """
    Fetch high liquidity coins from Binance USD-M futures with market cap over the specified threshold
    and update config.py with the results.
    
    Args:
        min_market_cap: Minimum market cap in millions of USD (default: 200 million)
        max_coins: Maximum number of coins to include (default: None - include all)
        timeframe: Timeframe to use for trading (default: "1m")
    """
    print(f"\nFetching coins with market cap > ${min_market_cap}M...")
    
    try:
        # Step 1: Get Binance USD-M perpetual futures symbols
        binance_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        binance_data = requests.get(binance_url).json()
        binance_symbols = [
            s['symbol'] for s in binance_data['symbols']
            if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT'
        ]
        print(f"Found {len(binance_symbols)} perpetual futures symbols on Binance")
        
        # Step 2: Get market caps from CoinGecko
        coingecko_url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 250,
            'page': 1
        }
        
        market_data = requests.get(coingecko_url, params=params).json()
        high_cap_coins = {
            coin['symbol'].upper(): coin['market_cap'] 
            for coin in market_data 
            if coin['market_cap'] >= min_market_cap * 1_000_000
        }
        
        print(f"Found {len(high_cap_coins)} coins with market cap >= ${min_market_cap}M")
        
        # Step 3: Filter Binance symbols by market cap
        filtered_symbols = []
        for symbol in binance_symbols:
            base = symbol.replace("USDT", "")  # Remove USDT to match CoinGecko format
            if base in high_cap_coins:
                filtered_symbols.append(symbol)
                
        # Sort by market cap (descending)
        filtered_symbols.sort(
            key=lambda x: high_cap_coins.get(x.replace("USDT", ""), 0),
            reverse=True
        )
        
        # Apply max_coins limit if specified
        if max_coins and int(max_coins) > 0:
            filtered_symbols = filtered_symbols[:int(max_coins)]
            print(f"Limited to top {len(filtered_symbols)} coins")
            
        if not filtered_symbols:
            print("No matching coins found, using default list")
            filtered_symbols = list(config.DEFAULT_UNIVERSE)
        
        # Step 4: Update config.py
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
        with open(config_path, 'r') as file:
            config_content = file.read()
        
        # Update PRIMARY_TIMEFRAME
        timeframe_pattern = r'PRIMARY_TIMEFRAME\s*=\s*".*?"'
        config_content = re.sub(
            timeframe_pattern,
            f'PRIMARY_TIMEFRAME = "{timeframe}"',
            config_content,
            count=1,
        )
        
        # Replace DEFAULT_UNIVERSE block
        universe_marker = "DEFAULT_UNIVERSE = ["
        start_idx = config_content.find(universe_marker)
        if start_idx == -1:
            print("Could not find DEFAULT_UNIVERSE section in config.py")
            return
        
        section_start = start_idx
        section_end = config_content.find("]", section_start)
        if section_end == -1:
            print("Could not determine end of DEFAULT_UNIVERSE list")
            return
        section_end += 1  # include closing bracket
        
        universe_lines = ["DEFAULT_UNIVERSE = ["]
        for symbol in filtered_symbols:
            universe_lines.append(f'    "{symbol}",')
        if filtered_symbols:
            universe_lines[-1] = universe_lines[-1].rstrip(",")  # remove trailing comma
        universe_lines.append("]")
        universe_block = "\n".join(universe_lines) + "\n"
        universe_block = "\n".join(universe_lines)
        
        config_content = (
            config_content[:section_start] +
            universe_block +
            config_content[section_end:]
        )
        
        with open(config_path, 'w') as file:
            file.write(config_content)
            
        print(f"\nUpdated config.py with {len(filtered_symbols)} high liquidity coins")
        
        # Print the list of selected coins
        print("\nSelected coins:")
        print("-" * 40)
        for i, symbol in enumerate(filtered_symbols, 1):
            print(f"{i:2d}. {symbol:10s} (Timeframe: {timeframe})")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to update config.py with high liquidity coins")

if __name__ == "__main__":
    # Default values
    min_cap = 500  # $500M
    max_coins = None
    timeframe = config.PRIMARY_TIMEFRAME
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        min_cap = int(sys.argv[1])
        
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        max_coins = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        timeframe = sys.argv[3]
    
    # Update config with high liquidity coins
    update_config_with_high_liquidity_coins(min_cap, max_coins, timeframe)
