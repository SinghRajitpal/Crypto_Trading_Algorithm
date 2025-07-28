import ccxt.pro as ccxt
import time
import config
import asyncio
from utils.logging_config import get_logger, console_log

# Get logger for this module
logger = get_logger(__name__)

class BinanceClient:
    """
    A client for Binance Futures trading via ccxt.pro.

    Needs a REVISION !
    
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
        
        # Configure exchange options for futures trading
        self.exchange.options['warnOnFetchOpenOrdersWithoutSymbol'] = False  # Suppress CCXT warning
        self.exchange.options['defaultType'] = 'future'  # Use futures API
        
        # Set sandbox mode if testnet is enabled
        if testnet:
            self.exchange.set_sandbox_mode(True)
            
    async def setup_account_config(self):
        """Setup account configuration for futures trading."""
        try:
            # Set position mode to one-way (easier to manage)
            # This avoids the "position side does not match" error
            await self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'false'})
            logger.info("Position mode set to One-Way")
        except Exception as e:
            # This might fail if already set, which is fine
            logger.warning(f"Position mode setting: {e}")
    
    # ===== Account & Position Info =====
    
    async def get_balance(self):
        """Get balance information for all assets"""
        return await self.exchange.fetch_balance()
    
    async def get_open_positions(self, symbol=None):
        """Get all open positions or filter by symbol"""
        try:
            # Store original symbol for logging
            original_symbol = symbol
            
            # Format symbol for API call
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
                if original_symbol:
                    pos = open_positions[0]
                    # Keep print for immediate feedback
                    print(f"\n[{original_symbol}] 📈 Active Position:")
                    print(f"Size: {pos['contracts']} contracts")
                    print(f"Entry Price: {pos.get('entryPrice', 'N/A')}")
                    print(f"Unrealized PnL: {pos.get('unrealizedPnl', 'N/A')}")
                    # Add detailed logging
                    logger.info(f"Active position for {original_symbol}: size={pos['contracts']}, entry={pos.get('entryPrice', 'N/A')}, pnl={pos.get('unrealizedPnl', 'N/A')}")
                else:
                    print(f"\n📈 Found {len(open_positions)} active position(s)")
                    logger.info(f"Found {len(open_positions)} active position(s)")
                
            return open_positions
        except Exception as e:
            print(f"\n❌ Error fetching positions: {e}")
            logger.error(f"Error fetching positions: {e}", exc_info=True)
            return []
    
    async def get_all_positions(self):
        """Get all open positions (alias for get_open_positions with no symbol filter)"""
        return await self.get_open_positions(symbol=None)
    
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
    
    async def monitor_orders(self, symbol: str, check_interval: int = 30) -> None:
        """Monitor orders for a symbol and cancel remaining orders when SL/TP is hit.
        
        Args:
            symbol: Symbol to monitor orders for.
            check_interval: How often to check orders in seconds.
        """
        try:
            logger.info(f"Starting order monitoring for {symbol}")
            
            while True:
                try:
                    # Get current positions
                    positions = await self.get_open_positions()
                    active_position = None
                    
                    # Check if we still have an active position for this symbol
                    for pos in positions:
                        if pos['symbol'] == symbol and abs(float(pos['size'])) > 0:
                            active_position = pos
                            break
                    
                    # If no active position, cancel any remaining orders
                    if not active_position:
                        logger.info(f"No active position for {symbol}, canceling remaining orders")
                        try:
                            cancelled_orders = await self.cancel_all_orders(symbol)
                            if cancelled_orders:
                                logger.success(f"Cancelled {len(cancelled_orders)} remaining orders for {symbol}")
                            break  # Exit monitoring for this symbol
                        except Exception as e:
                            logger.warning(f"Error cancelling orders: {e}")
                    
                    # Check open orders
                    open_orders = await self.exchange.fetch_open_orders(symbol)
                    
                    if not open_orders:
                        if active_position:
                            logger.debug(f"No open orders for {symbol} but position exists")
                        else:
                            logger.debug(f"No orders to monitor for {symbol}")
                            break
                    
                    # Log current order status
                    stop_loss_orders = [o for o in open_orders if o['type'] == 'stop']
                    take_profit_orders = [o for o in open_orders if 'take_profit' in o['type'].lower()]
                    
                    if stop_loss_orders or take_profit_orders:
                        logger.debug(f"Monitoring {len(stop_loss_orders)} SL and {len(take_profit_orders)} TP orders for {symbol}")
                    
                    await asyncio.sleep(check_interval)
                    
                except Exception as e:
                    logger.warning(f"Error in order monitoring loop: {e}")
                    await asyncio.sleep(check_interval)
                    
        except asyncio.CancelledError:
            logger.info(f"Order monitoring cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Order monitoring error for {symbol}: {e}", exc_info=True)

    async def close_position(self, symbol, side=None):
        """Close an open position for a symbol"""
        # Import debug settings
        import config
        debug_settings = getattr(config, 'debug', {})
        verbose = debug_settings.get('verbose_logging', False)
        
        # Track original symbol for logging
        original_symbol = symbol
        
        # Ensure symbol is correctly formatted
        symbol = self._format_symbol(symbol)
        
        # Get existing positions
        positions = await self.get_open_positions(original_symbol)
        
        if not positions:
            logger.warning(f"No positions found for {original_symbol}")
            return {"status": "no_position", "symbol": original_symbol}
        
        # First, cancel any open orders for this symbol
        try:
            logger.info(f"Canceling any open orders for {original_symbol}")
            await self.exchange.cancel_all_orders(symbol)
        except Exception as e:
            logger.warning(f"Failed to cancel open orders: {e}")
            # Continue anyway to close the position
        
        orders = []
        
        # Check if we're in hedge mode
        try:
            # Try to get position mode - use the correct CCXT method
            position_mode = await self.exchange.fapiPrivateGetPositionSideDual()
            is_hedge_mode = position_mode.get('dualSidePosition', False)
            logger.debug(f"Hedge mode is {'enabled' if is_hedge_mode else 'disabled'}")
        except Exception as e:
            logger.debug(f"Error checking hedge mode: {e}")
            is_hedge_mode = False  # Assume one-way mode by default for testnet
        
        for position in positions:
            contracts = float(position.get('contracts', 0))
            
            if contracts == 0:
                continue
                
            position_side = 'long' if contracts > 0 else 'short'
            
            # Skip if we're only closing one side and this isn't it
            if side and position_side != side:
                continue
                
            # Determine the appropriate side for closing
            close_side = 'sell' if position_side == 'long' else 'buy'
            amount = abs(contracts)
            
            # Look for hedge mode position side in the position data
            hedge_position_side = None
            if is_hedge_mode:
                if 'positionSide' in position:
                    hedge_position_side = position['positionSide']
                else:
                    hedge_position_side = 'LONG' if position_side == 'long' else 'SHORT'
                logger.debug(f"Position side: {hedge_position_side}")
            
            if amount > 0:
                try:
                    logger.info(f"Closing {position_side.upper()} position for {original_symbol}")
                    logger.debug(f"Position size: {amount:.6f} contracts")
                    
                    # Use reduce only for safety
                    params = {
                        'reduceOnly': True
                    }
                    
                    # Add position side if in hedge mode
                    if is_hedge_mode and hedge_position_side:
                        params['positionSide'] = hedge_position_side
                    
                    order = await self.exchange.create_market_order(
                        symbol=symbol,
                        side=close_side,
                        amount=amount,
                        params=params
                    )
                    
                    orders.append(order)
                    logger.success(f"Successfully closed position with order: {order.get('id', 'Unknown')}")
                
                except Exception as e:
                    error_str = str(e)
                    logger.error(f"Error closing position: {error_str}")
                    
                    # Try alternative methods if first attempt fails
                    if "close position" in error_str.lower() or "reduce" in error_str.lower():
                        try:
                            logger.info("Trying alternative closing method...")
                            
                            # Try using CLOSE_POSITION which doesn't need amount
                            close_params = {
                                'closePosition': True
                            }
                            
                            # Add position side if in hedge mode
                            if is_hedge_mode and hedge_position_side:
                                close_params['positionSide'] = hedge_position_side
                            
                            order = await self.exchange.create_market_order(
                                symbol=symbol,
                                side=close_side,
                                amount=0,  # Not needed with closePosition
                                params=close_params
                            )
                            
                            orders.append(order)
                            logger.success("Successfully closed position with alternative method")
                        
                        except Exception as alt_e:
                            alt_error = str(alt_e)
                            logger.error(f"Alternative closing method also failed: {alt_error}")
                            
                            # Last resort: try a third method
                            try:
                                logger.info("Trying third closing method (exact amount)...")
                                
                                # Try getting more precise position size
                                exact_amount = float(position.get('contracts', amount))
                                
                                order = await self.exchange.create_market_order(
                                    symbol=symbol,
                                    side=close_side,
                                    amount=exact_amount
                                )
                                
                                orders.append(order)
                                logger.success("Successfully closed position with third method")
                                
                            except Exception as third_e:
                                logger.critical(f"All closing methods failed: {str(third_e)}")
                                return {"status": "error", "symbol": original_symbol, 
                                       "error": f"All closing methods failed: {str(third_e)}"}
                    else:
                        return {"status": "error", "symbol": original_symbol, "error": error_str}
        
        if not orders:
            if side:
                return {"status": "no_matching_position", "symbol": original_symbol, "side": side}
            else:
                return {"status": "no_position", "symbol": original_symbol}
        
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
    
    
    
    async def open_position(self, symbol: str, side: str, amount: float, price=None, stop_loss=None, take_profit=None, leverage=None, margin_type=None):
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
            # Import debug settings
            import config
            debug_settings = getattr(config, 'debug', {})
            verbose = debug_settings.get('verbose_logging', False)
            
            # Track original symbol for logging
            original_symbol = symbol
            
            # Ensure symbol is correctly formatted
            symbol = self._format_symbol(symbol)
            
            print(f"\n[Binance] Opening {side.upper()} position for {original_symbol}")
            print(f"[Binance] Details: Size={amount:.6f}, Leverage={leverage}x")
            if stop_loss is not None:
                print(f"[Binance] Stop Loss: {stop_loss:.2f}")
            if take_profit is not None:
                print(f"[Binance] Take Profit: {take_profit:.2f}")
            
            # Log detailed information
            logger.info(f"Opening {side.upper()} position for {original_symbol}: size={amount:.6f}, leverage={leverage}x, sl={stop_loss}, tp={take_profit}")
            
            # Check if we're in hedge mode
            try:
                position_mode = await self.exchange.fapiPrivateGetPositionSideDual()
                is_hedge_mode = position_mode.get('dualSidePosition', False)
                logger.debug(f"Hedge mode is {'enabled' if is_hedge_mode else 'disabled'}")
            except Exception as e:
                logger.debug(f"Error checking hedge mode: {e}")
                is_hedge_mode = False  # Assume one-way mode by default for testnet
            
            # Determine position side based on side
            position_side = 'LONG' if side == 'buy' else 'SHORT'
            logger.debug(f"Position side: {position_side}")
                
            # Set leverage if provided
            if leverage is not None:
                try:
                    leverage_result = await self.set_leverage(symbol, leverage)
                    logger.info(f"Leverage set to {leverage}x")
                except Exception as e:
                    logger.warning(f"Could not set leverage to {leverage}x: {e}")
                    # Continue with default leverage
                
            # Set margin type if provided
            if margin_type is not None:
                try:
                    margin_result = await self.set_margin_type(symbol, margin_type)
                    logger.info(f"Margin type set to {margin_type}")
                except Exception as e:
                    logger.warning(f"Could not set margin type to {margin_type}: {e}")
                    # Continue with default margin type
                
            # Determine order type based on price
            order_type = 'limit' if price is not None else 'market'
            
            # Prepare parameters for the main order
            params = {}
            
            # Add position side for hedge mode
            if is_hedge_mode:
                params['positionSide'] = position_side
            
            logger.info(f"Sending {order_type} {side} order to exchange...")
            
            # Calculate estimated notional value
            estimated_notional = amount * (price if price is not None else 
                                          (await self.exchange.fetch_ticker(symbol))['last'])
            logger.debug(f"Estimated order notional value: ${estimated_notional:.2f}")
            
            # Check minimum notional requirement
            if estimated_notional < 100:
                error_msg = (f"Order notional value (${estimated_notional:.2f}) is below the minimum "
                             f"requirement of $100.00. Consider increasing position size or leverage.")
                logger.error(error_msg)
                return {
                    "status": "error",
                    "symbol": original_symbol,
                    "error": error_msg
                }
            
            try:
                # Place the main position order
                main_order = await self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price,
                    params=params
                )
                
                # Keep critical success message as print for immediate feedback
                print(f"[Binance] ✅ Main order successful!")
                print(f"[Binance] Order ID: {main_order.get('id', 'Unknown')}")
                print(f"[Binance] Order price: {main_order.get('price', 'Market price')}")
                
                # Add detailed logging
                logger.success(f"Main order successful for {original_symbol}: ID={main_order.get('id', 'Unknown')}, price={main_order.get('price', 'Market price')}")
                
            except Exception as e:
                error_str = str(e)
                # Handle common Binance errors with more helpful messages
                if "Order's notional must be no smaller than 100" in error_str:
                    error_msg = ("Order notional value too small. Binance requires a minimum order value of $100. "
                                "Try increasing position size or leverage.")
                elif "LOT_SIZE" in error_str:
                    error_msg = (f"Invalid lot size for {original_symbol}. The amount must be a multiple "
                                f"of the minimum lot size. Please adjust your position size.")
                elif "PRICE_FILTER" in error_str:
                    error_msg = (f"Invalid price for {original_symbol}. The price doesn't conform to "
                                f"the exchange's price filters. Try using a market order instead.")
                else:
                    error_msg = f"Error opening position: {error_str}"
                
                logger.error(error_msg)
                return {
                    "status": "error",
                    "symbol": original_symbol,
                    "error": error_msg
                }
            
            # Wait a moment for the order to be processed
            await asyncio.sleep(1)
            
            # Get the actual fill price for direction validation
            actual_entry_price = main_order.get('average') or main_order.get('price') or price
            if actual_entry_price is None:
                # Fetch current market price as fallback
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    actual_entry_price = ticker['last']
                except:
                    actual_entry_price = price  # Use original price as last resort
            
            # Calculate opposite side for closing orders
            close_side = 'sell' if side == 'buy' else 'buy'
            
            # Place stop loss order if price is provided
            stop_loss_order = None
            if stop_loss is not None:
                try:
                    # Use direct Binance API parameters for stop loss
                    print(f"[Binance] Creating stop loss order at price: {stop_loss:.2f}")
                    
                    # For Binance Futures, we need to use specific parameter format
                    stop_loss_params = {
                        'stopPrice': stop_loss,
                    }
                    
                    # Add position side if in hedge mode
                    if is_hedge_mode:
                        stop_loss_params['positionSide'] = position_side
                    
                    stop_loss_order = await self.exchange.create_order(
                        symbol=symbol,
                        type='STOP_MARKET',
                        side=close_side,
                        amount=amount,
                        params=stop_loss_params
                    )
                    
                    print(f"[Binance] ✅ Stop loss order created: {stop_loss_order.get('id', 'Unknown')}")
                    
                except Exception as e:
                    print(f"[Binance] ⚠️ Failed to create stop loss order: {e}")
                    
                    # Try alternative method directly with Binance's API structure
                    try:
                        print(f"[Binance] Attempting alternative stop loss method...")
                        # Use closePosition parameter for simplicity
                        stop_loss_params = {
                            'closePosition': 'true',
                            'stopPrice': stop_loss,
                        }
                        
                        # Add position side if in hedge mode
                        if is_hedge_mode:
                            stop_loss_params['positionSide'] = position_side
                        
                        stop_loss_order = await self.exchange.create_order(
                            symbol=symbol,
                            type='STOP_MARKET',
                            side=close_side,
                            amount=0,  # Amount is not needed with closePosition=true
                            params=stop_loss_params
                        )
                        
                        print(f"[Binance] ✅ Stop loss order created with alternative method: {stop_loss_order.get('id', 'Unknown')}")
                    except Exception as alt_e:
                        print(f"[Binance] ⚠️ Failed to create stop loss with alternative method: {alt_e}")
                        # Continue with the main order even if SL failed
            
            # Place take profit order if price is provided
            take_profit_order = None
            if take_profit is not None:
                try:
                    # Validate take profit direction first
                    if ((side == 'buy' and take_profit <= actual_entry_price) or 
                        (side == 'sell' and take_profit >= actual_entry_price)):
                        print(f"[Binance] ⚠️ Invalid take profit direction: TP={take_profit:.2f}, Entry={actual_entry_price:.2f}, Side={side}")
                        print(f"[Binance] Correcting take profit direction...")
                        # Skip take profit if direction is wrong to prevent losses
                        take_profit = None
                    
                    if take_profit is not None:
                        print(f"[Binance] Creating take profit order at price: {take_profit:.2f} (Entry: {actual_entry_price:.2f}, Side: {side})")
                        
                        # For Binance Futures TAKE_PROFIT_MARKET orders
                        take_profit_params = {
                            'stopPrice': take_profit,
                            'timeInForce': 'GTC',  # Good Till Cancelled
                        }
                        
                        # Add position side if in hedge mode
                        if is_hedge_mode:
                            take_profit_params['positionSide'] = position_side
                        
                        take_profit_order = await self.exchange.create_order(
                            symbol=symbol,
                            type='TAKE_PROFIT_MARKET',
                            side=close_side,
                            amount=amount,
                            params=take_profit_params
                        )
                        
                        print(f"[Binance] ✅ Take profit order created: {take_profit_order.get('id', 'Unknown')}")
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"[Binance] ⚠️ Failed to create take profit order: {e}")
                    
                    # Try alternative method for problematic cases
                    if take_profit is not None and ('invalid' not in error_msg and 'format' not in error_msg):
                        try:
                            print(f"[Binance] Attempting simplified take profit method...")
                            # Simplified approach with basic parameters
                            take_profit_params = {
                                'stopPrice': take_profit,
                            }
                            
                            if is_hedge_mode:
                                take_profit_params['positionSide'] = position_side
                            
                            take_profit_order = await self.exchange.create_order(
                                symbol=symbol,
                                type='TAKE_PROFIT_MARKET',
                                side=close_side,
                                amount=amount,
                                params=take_profit_params
                            )
                            
                            print(f"[Binance] ✅ Take profit order created with simplified method: {take_profit_order.get('id', 'Unknown')}")
                        except Exception as alt_e:
                            print(f"[Binance] ⚠️ Both take profit methods failed: {alt_e}")
                            # Continue with the main order even if TP failed
            
            # Start order monitoring in background if we have SL or TP orders
            if stop_loss_order is not None or take_profit_order is not None:
                try:
                    # Start monitoring task in background
                    asyncio.create_task(self.monitor_orders(symbol, check_interval=30))
                    print(f"[Binance] ✅ Started order monitoring for {original_symbol}")
                except Exception as monitor_e:
                    print(f"[Binance] ⚠️ Failed to start order monitoring: {monitor_e}")
            
            # Prepare the result with all order information
            return {
                "status": "success",
                "order": main_order,
                "position": {
                    "symbol": original_symbol,
                    "side": side,
                    "size": amount,
                    "leverage": leverage,
                    "margin_type": margin_type,
                    "stop_loss": stop_loss_order,
                    "take_profit": take_profit_order
                }
            }
        except Exception as e:
            error_message = f"Error opening position: {str(e)}"
            logger.error(error_message, exc_info=True)
            
            return {
                "status": "error",
                "symbol": symbol,
                "error": error_message
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
        # If symbol already contains '/' or is None, return as is
        if not symbol:
            return symbol
            
        if '/' in symbol:
            return symbol
            
        # Import debug settings
        import config
        debug_settings = getattr(config, 'debug', {})
        verbose = debug_settings.get('verbose_logging', False)
        print_format = debug_settings.get('print_symbol_format', False)
        
        original_symbol = symbol
        formatted_symbol = None
            
        # For USDT pairs (most common)
        if symbol.endswith('USDT'):
            base = symbol[:-4]  # Remove USDT
            quote = 'USDT'
            formatted_symbol = f"{base}/{quote}"
            
        # For BTC pairs
        elif symbol.endswith('BTC'):
            base = symbol[:-3]  # Remove BTC
            quote = 'BTC'
            formatted_symbol = f"{base}/{quote}"
            
        # For ETH pairs
        elif symbol.endswith('ETH'):
            base = symbol[:-3]  # Remove ETH
            quote = 'ETH'
            formatted_symbol = f"{base}/{quote}"
            
        # For BUSD pairs
        elif symbol.endswith('BUSD'):
            base = symbol[:-4]  # Remove BUSD
            quote = 'BUSD'
            formatted_symbol = f"{base}/{quote}"
        
        # Default case - try to detect quote currency or use USDT as default
        else:
            # Common quote currencies and their lengths
            quote_currencies = {
                'USDT': 4,
                'BUSD': 4,
                'USDC': 4,
                'BTC': 3,
                'ETH': 3,
                'BNB': 3,
                'USD': 3,
                'EUR': 3
            }
            
            # Try to find a matching quote currency
            for quote, length in quote_currencies.items():
                if len(symbol) > length and symbol.endswith(quote):
                    base = symbol[:-length]
                    formatted_symbol = f"{base}/{quote}"
                    break
            
            # If we can't determine, default to symbol/USDT
            if not formatted_symbol:
                logger.warning(f"Couldn't parse symbol format for {symbol}, using {symbol}/USDT")
                formatted_symbol = f"{symbol}/USDT"
        
        # Print symbol format conversion if enabled (for debugging)
        if print_format:
            logger.debug(f"Symbol format: {original_symbol} -> {formatted_symbol}")
            
        return formatted_symbol
    
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
            logger.info("Exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")