"""
Order Management Module

Handles stop loss and take profit order monitoring and cancellation.
When one order fills, the opposing order should be cancelled.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class TrackedOrder:
    """Represents a tracked order with its associated orders."""
    order_id: str
    symbol: str
    side: str
    order_type: str  # 'main', 'stop_loss', 'take_profit'
    price: float
    amount: float
    status: str
    associated_orders: List[str]  # IDs of related SL/TP orders
    timestamp: float

class OrderManager:
    """Manages order execution, tracking, and automatic SL/TP cancellation."""
    
    def __init__(self, binance_client):
        """Initialize the order manager.
        
        Args:
            binance_client: Binance client for order operations.
        """
        self.binance_client = binance_client
        self.tracked_orders: Dict[str, TrackedOrder] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start_monitoring(self):
        """Start the order monitoring task."""
        if self.monitoring_task:
            return
            
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitor_orders())
        print("[OrderManager] Started order monitoring")
    
    async def stop_monitoring(self):
        """Stop the order monitoring task."""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        print("[OrderManager] Stopped order monitoring")
    
    async def place_position_with_sltp(self, symbol: str, side: str, amount: float,
                                      stop_loss: Optional[float] = None,
                                      take_profit: Optional[float] = None,
                                      leverage: int = 1) -> Dict[str, Any]:
        """Place a position with integrated stop loss and take profit orders.
        
        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            amount: Position size.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            leverage: Leverage to use.
            
        Returns:
            Dictionary with order results.
        """
        try:
            print(f"[OrderManager] Placing {side.upper()} position for {symbol}")
            
            # Determine position side for hedge mode
            position_side = 'LONG' if side == 'buy' else 'SHORT'
            
            # Place the main market order with position side
            main_order = await self.binance_client.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params={'positionSide': position_side}
            )
            
            main_order_id = main_order['id']
            associated_orders = []
            
            print(f"[OrderManager] ✅ Main order placed: {main_order_id}")
            
            # Place stop loss order if specified
            sl_order_id = None
            if stop_loss:
                try:
                    sl_side = 'sell' if side == 'buy' else 'buy'  # Opposite side
                    sl_order = await self.binance_client.exchange.create_order(
                        symbol=symbol,
                        type='stop_market',
                        side=sl_side,
                        amount=amount,
                        params={
                            'stopPrice': stop_loss,
                            'positionSide': position_side  # Same position side
                        }
                    )
                    sl_order_id = sl_order['id']
                    associated_orders.append(sl_order_id)
                    print(f"[OrderManager] ✅ Stop loss order placed: {sl_order_id} @ {stop_loss}")
                except Exception as e:
                    print(f"[OrderManager] ❌ Failed to place stop loss: {e}")
            
            # Place take profit order if specified
            tp_order_id = None
            if take_profit:
                try:
                    tp_side = 'sell' if side == 'buy' else 'buy'  # Opposite side
                    # Use take profit market order for proper take profit execution
                    tp_order = await self.binance_client.exchange.create_order(
                        symbol=symbol,
                        type='take_profit_market',  # Proper take profit market order
                        side=tp_side,
                        amount=amount,
                        params={
                            'stopPrice': take_profit,
                            'positionSide': position_side  # Same position side
                        }
                    )
                    tp_order_id = tp_order['id']
                    associated_orders.append(tp_order_id)
                    print(f"[OrderManager] ✅ Take profit market order placed: {tp_order_id} @ {take_profit}")
                except Exception as e:
                    print(f"[OrderManager] ❌ Failed to place take profit: {e}")
            
            # Track the main order and associated SL/TP orders
            current_time = time.time()
            
            # Track main order
            self.tracked_orders[main_order_id] = TrackedOrder(
                order_id=main_order_id,
                symbol=symbol,
                side=side,
                order_type='main',
                price=main_order.get('price', 0),
                amount=amount,
                status='filled',  # Market orders are immediately filled
                associated_orders=associated_orders,
                timestamp=current_time
            )
            
            # Track SL order
            if sl_order_id:
                self.tracked_orders[sl_order_id] = TrackedOrder(
                    order_id=sl_order_id,
                    symbol=symbol,
                    side=sl_side,
                    order_type='stop_loss',
                    price=stop_loss,
                    amount=amount,
                    status='open',
                    associated_orders=[tp_order_id] if tp_order_id else [],
                    timestamp=current_time
                )
            
            # Track TP order
            if tp_order_id:
                self.tracked_orders[tp_order_id] = TrackedOrder(
                    order_id=tp_order_id,
                    symbol=symbol,
                    side=tp_side,
                    order_type='take_profit',
                    price=take_profit,
                    amount=amount,
                    status='open',
                    associated_orders=[sl_order_id] if sl_order_id else [],
                    timestamp=current_time
                )
            
            return {
                'status': 'success',
                'main_order_id': main_order_id,
                'stop_loss_order_id': sl_order_id,
                'take_profit_order_id': tp_order_id,
                'associated_orders': associated_orders
            }
            
        except Exception as e:
            print(f"[OrderManager] ❌ Failed to place position: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _monitor_orders(self):
        """Monitor tracked orders and handle SL/TP cancellation."""
        while self.running:
            try:
                # Get all tracked orders that are still open
                open_orders = [order for order in self.tracked_orders.values() 
                              if order.status == 'open']
                
                if not open_orders:
                    await asyncio.sleep(5)  # Check every 5 seconds
                    continue
                
                # Check status of each open order
                for tracked_order in open_orders:
                    try:
                        # Fetch current order status
                        order_status = await self.binance_client.exchange.fetch_order(
                            tracked_order.order_id, 
                            tracked_order.symbol
                        )
                        
                        current_status = order_status.get('status', 'unknown')
                        
                        # If order was filled, cancel associated orders
                        if current_status == 'filled' and tracked_order.status == 'open':
                            print(f"[OrderManager] 🎯 {tracked_order.order_type.upper()} order filled: {tracked_order.order_id}")
                            tracked_order.status = 'filled'
                            
                            # Cancel associated orders
                            await self._cancel_associated_orders(tracked_order)
                        
                        # Update order status
                        elif current_status != tracked_order.status:
                            tracked_order.status = current_status
                            
                    except Exception as e:
                        print(f"[OrderManager] Error checking order {tracked_order.order_id}: {e}")
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[OrderManager] Error in order monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _cancel_associated_orders(self, filled_order: TrackedOrder):
        """Cancel orders associated with a filled order."""
        for order_id in filled_order.associated_orders:
            if order_id in self.tracked_orders:
                associated_order = self.tracked_orders[order_id]
                
                # Only cancel if still open
                if associated_order.status == 'open':
                    try:
                        await self.binance_client.exchange.cancel_order(
                            order_id, 
                            associated_order.symbol
                        )
                        associated_order.status = 'cancelled'
                        print(f"[OrderManager] ✅ Cancelled {associated_order.order_type} order: {order_id}")
                    except Exception as e:
                        print(f"[OrderManager] ❌ Failed to cancel order {order_id}: {e}")
    
    def get_tracked_orders(self) -> Dict[str, TrackedOrder]:
        """Get all tracked orders."""
        return self.tracked_orders.copy()
    
    def get_active_positions(self) -> List[str]:
        """Get symbols with active positions (filled main orders with pending SL/TP)."""
        active_symbols = set()
        
        for order in self.tracked_orders.values():
            if (order.order_type == 'main' and order.status == 'filled' and 
                any(self.tracked_orders.get(assoc_id, {}).status == 'open' 
                    for assoc_id in order.associated_orders)):
                active_symbols.add(order.symbol)
        
        return list(active_symbols)
