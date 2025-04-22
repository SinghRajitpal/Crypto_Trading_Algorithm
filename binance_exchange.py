import ccxt.pro as ccxt
import time
import config

class BinanceClient:
    """
    A professional client for Binance Futures trading via ccxt.pro.
    
    Provides access to:
    - Account and position management
    - Order execution with integrated risk parameters
    - Leverage and margin control
    - Position metrics and risk assessment
    
    All operations use REST API for reliable execution.
    """
    
    def __init__(self, testnet=True):
        """
        Initialize the Binance client
        
        Args:
            testnet: Whether to use testnet (default: True)
        """
        # Common options
        options = {
            'enableRateLimit': True,
        }
        
        # Set API credentials based on testnet setting
        if testnet:
            api_key = config.binance_futures_testnet.get('testnet_api_key')
            api_secret = config.binance_futures_testnet.get('testnet_api_secret')
            options['testnet'] = True
        else:
            api_key = config.binance_futures.get('api_key')
            api_secret = config.binance_futures.get('api_secret')
        
        if api_key and api_secret:
            options['apiKey'] = api_key
            options['secret'] = api_secret
        
        # Initialize exchange
        self.exchange = ccxt.binanceusdm(options)
        
        # Set sandbox mode if testnet is enabled
        if testnet:
            self.exchange.set_sandbox_mode(True)
    
    # ===== Account & Position Info =====
    
    async def get_balance(self):
        """Get balance information for all assets"""
        return await self.exchange.fetch_balance()
    
    async def get_open_positions(self, symbol=None):
        """Get all open positions or filter by symbol"""
        try:
            formatted_symbol = self._format_symbol(symbol) if symbol else None
            if formatted_symbol:
                positions = await self.exchange.fetch_positions([formatted_symbol])
            else:
                positions = await self.exchange.fetch_positions()
                
            # Filter for non-zero positions and convert symbols back to standard format
            open_positions = []
            for p in positions:
                if float(p['contracts']) != 0:
                    p['symbol'] = self._unformat_symbol(p['symbol'])
                    open_positions.append(p)
            
            # Only log when there are actual positions
            if open_positions:
                if symbol:
                    pos = open_positions[0]
                    print(f"\n[{symbol}] 📈 Active Position:")
                    print(f"Size: {pos['contracts']} contracts")
                    print(f"Entry Price: {pos.get('entryPrice', 'N/A')}")
                    print(f"Unrealized PnL: {pos.get('unrealizedPnl', 'N/A')}")
                else:
                    print(f"\n📈 Found {len(open_positions)} active position(s)")
                
            return open_positions
        except Exception as e:
            print(f"\n❌ Error fetching positions: {e}")
            return []
    
    async def get_leverage(self, symbol):
        """Get current leverage for a specific symbol"""
        symbol = self._format_symbol(symbol)
        positions = await self.get_open_positions(symbol)
        if positions:
            return int(positions[0]['leverage'])
        else:
            leverage_info = await self.exchange.fetch_leverage(symbol)
            return int(leverage_info['leverage'])
    
    async def get_margin_type(self, symbol):
        """Get current margin type for a specific symbol"""
        symbol = self._format_symbol(symbol)
        positions = await self.get_open_positions(symbol)
        if positions:
            return positions[0]['marginType'].lower()
        else:
            margin_info = await self.exchange.fetch_position_mode(symbol)
            return 'cross' if margin_info.get('crossMargin') else 'isolated'
    
    async def get_account_metrics(self):
        """Get comprehensive account metrics for risk management"""
        balance = await self.get_balance()
        positions = await self.get_open_positions()
        
        total_wallet_balance = float(balance['total']['USDT']) if 'USDT' in balance['total'] else 0
        total_unrealized_pnl = sum(float(p['unrealizedPnl']) for p in positions)
        total_margin_used = sum(float(p['initialMargin']) for p in positions)
        available_margin = total_wallet_balance - total_margin_used
        exposure_percentage = (total_margin_used / total_wallet_balance * 100) if total_wallet_balance else 0
        
        return {
            'total_wallet_balance': total_wallet_balance,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_margin_used': total_margin_used,
            'available_margin': available_margin,
            'exposure_percentage': exposure_percentage,
            'position_count': len(positions)
        }
    
    async def get_liquidation_data(self, symbol):
        """Get liquidation price data for a position"""
        positions = await self.get_open_positions(symbol)
        if not positions:
            return {"liquidationPrice": None, "marginType": None, "leverage": None}
        return positions[0]
    
    # ===== Execution Layer =====
    
    async def create_order(self, symbol, order_type, side, amount, price=None, params={}):
        """Create a new order"""
        symbol = self._format_symbol(symbol)
        return await self.exchange.create_order(symbol, order_type, side, amount, price, params)
    
    async def cancel_order(self, order_id, symbol):
        """Cancel an existing order"""
        symbol = self._format_symbol(symbol)
        return await self.exchange.cancel_order(order_id, symbol)
    
    async def get_open_orders(self, symbol=None):
        """Get all open orders, optionally filtered by symbol"""
        if symbol:
            symbol = self._format_symbol(symbol)
        return await self.exchange.fetch_open_orders(symbol)
    
    async def cancel_all_orders(self, symbol=None):
        """
        Cancel all open orders for a symbol or all symbols
        
        Args:
            symbol: Symbol to cancel orders for (None for all symbols)
            
        Returns:
            List of cancelled orders
        """
        try:
            if symbol:
                symbol = self._format_symbol(symbol)
                return await self.exchange.cancel_all_orders(symbol)
            else:
                # Get all open orders
                open_orders = await self.get_open_orders()
                
                # Group orders by symbol
                orders_by_symbol = {}
                for order in open_orders:
                    if order['symbol'] not in orders_by_symbol:
                        orders_by_symbol[order['symbol']] = []
                    orders_by_symbol[order['symbol']].append(order)
                
                # Cancel orders for each symbol
                results = []
                for symbol, orders in orders_by_symbol.items():
                    result = await self.exchange.cancel_all_orders(symbol)
                    results.extend(result if isinstance(result, list) else [result])
                
                return results
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    # ===== Position Management =====
    
    async def close_position(self, symbol, side=None):
        """Close an open position for a symbol"""
        symbol = self._format_symbol(symbol)
        positions = await self.get_open_positions(symbol)
        
        if not positions:
            return {"status": "no_position", "symbol": symbol}
        
        orders = []
        for position in positions:
            position_side = 'long' if float(position['contracts']) > 0 else 'short'
            
            if side and position_side != side:
                continue
                
            close_side = 'sell' if position_side == 'long' else 'buy'
            amount = abs(float(position['contracts']))
            
            if amount > 0:
                try:
                    order = await self.exchange.create_order(
                        symbol=symbol,
                        order_type='market',
                        side=close_side,
                        amount=amount,
                        params={'reduceOnly': True}
                    )
                    orders.append(order)
                except Exception as e:
                    return {"status": "error", "symbol": symbol, "error": str(e)}
        
        if not orders:
            return {"status": "no_matching_position", "symbol": symbol, "side": side}
        
        return {"status": "closed", "orders": orders}
    
    async def set_leverage(self, symbol, leverage):
        """Set leverage for a specific symbol"""
        symbol = self._format_symbol(symbol)
        return await self.exchange.set_leverage(leverage, symbol)
    
    async def set_margin_type(self, symbol, margin_type):
        """Set margin type for a specific symbol"""
        symbol = self._format_symbol(symbol)
        margin_type = margin_type.upper()
        return await self.exchange.set_margin_mode(margin_type, symbol)
    
    
    
    async def open_position(self, symbol, side, amount, price=None, stop_loss=None, take_profit=None, leverage=None, margin_type=None):
        """
        Open a position with integrated risk management
        
        This is a convenience method that combines setting leverage, margin type, and order creation
        with stop-loss and take-profit in a single call.
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' for long, 'sell' for short
            amount: Position size
            price: Entry price (None for market orders)
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            leverage: Leverage to use (e.g. 5x, 10x)
            margin_type: 'ISOLATED' or 'CROSS'
            
        Returns:
            Dictionary with order information
        """
        try:
            # Ensure symbol is correctly formatted
            symbol = self._format_symbol(symbol)
            
            # Set leverage if provided
            if leverage is not None:
                await self.set_leverage(symbol, leverage)
                
            # Set margin type if provided
            if margin_type is not None:
                await self.set_margin_type(symbol, margin_type)
                
            # Determine order type based on price
            order_type = 'limit' if price is not None else 'market'
            
            # Prepare parameters including risk management
            params = {}
            
            # Add stop loss to params if provided
            if stop_loss is not None:
                params['stopLoss'] = {'stopLossPrice': stop_loss}
            
            # Add take profit to params if provided
            if take_profit is not None:
                params['takeProfit'] = {'takeProfitPrice': take_profit}
                
            # Create the order with all parameters
            order = await self.exchange.create_order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            return {
                "status": "success",
                "order": order,
                "position": {
                    "symbol": symbol,
                    "side": side,
                    "size": amount,
                    "leverage": leverage,
                    "margin_type": margin_type,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "symbol": symbol,
                "error": str(e)
            }
    
    # ===== Helper Methods =====
    
    def _format_symbol(self, symbol):
        """
        Format symbol to ensure it matches exchange requirements.
        Handles both formats: with slash (BTC/USDT) and without (BTCUSDT).
        
        Args:
            symbol (str): Symbol in either format
            
        Returns:
            str: Symbol in exchange format (with slash)
        """
        # If symbol already contains '/', return as is
        if '/' in symbol:
            return symbol
            
        # Add slash before the quote currency
        if symbol.endswith('USDT'):
            return f"{symbol[:-4]}/USDT"
        elif symbol.endswith('BTC'):
            return f"{symbol[:-3]}/BTC"
        elif symbol.endswith('ETH'):
            return f"{symbol[:-3]}/ETH"
        else:
            # Default to USDT pair if we can't determine
            return f"{symbol}/USDT"
            
    def _unformat_symbol(self, symbol):
        """
        Convert exchange symbol format back to standard format.
        
        Args:
            symbol (str): Symbol with slash (e.g., "BTC/USDT")
            
        Returns:
            str: Symbol without slash (e.g., "BTCUSDT")
        """
        return symbol.replace('/', '') if symbol else ''
        
    async def close(self):
        """Close the exchange connection"""
        try:
            await self.exchange.close()
        except Exception as e:
            print(f"Error closing connection: {e}")