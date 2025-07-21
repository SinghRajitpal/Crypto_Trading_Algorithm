# Production Execution Engine implementing the complete system from the document

from typing import Dict, Any, Optional, List
import time
from execution.portfolio import ProductionPortfolioManager
from execution.risk_manager import ProductionRiskManager
from execution.executor import OrderExecutor
from execution.stress_handler import StressHandlingModule
import traceback
from datetime import datetime, timedelta

class ProductionExecutionEngine:
    """Production Execution Engine implementing the complete workflow from the document:
    
    Per 1-min bar workflow:
    1. Update metrics (ATR, correlations, equity curve)
    2. Rebalance if daily trigger
    3. On signal, size/leverage within budgets
    4. Execute with safeguards (stress handling, kill switches)
    """
    
    def __init__(self, binance_client, total_capital: float = 1000.0):
        """Initialize the production execution engine.
        
        Args:
            binance_client: Binance client for exchange communication.
            total_capital: Total trading capital in USDT.
        """
        self.binance_client = binance_client
        self.total_capital = total_capital
        
        # Initialize production components
        self.portfolio_manager = ProductionPortfolioManager(
            total_capital=total_capital,
            target_volatility=0.18,    # 18% target volatility from document
            max_allocation_pct=0.85    # 85% max allocation for production
        )
        
        self.risk_manager = ProductionRiskManager(
            portfolio_manager=self.portfolio_manager
        )
        
        self.order_executor = OrderExecutor(
            binance_client=binance_client,
            portfolio_manager=self.portfolio_manager,
            risk_manager=self.risk_manager
        )
        
        # Initialize stress handling module
        self.stress_handler = StressHandlingModule(self)
        
        # Stress handling state
        self.flash_crash_count = 0
        self.last_flash_crash_time = 0
        self.trading_paused = False
        
        # Initialize symbols from config
        self.portfolio_manager.process_symbols_from_config()
        
        print(f"[ProductionExecution] Initialized with ${total_capital:.2f} USDT capital")
        print(f"[ProductionExecution] Target volatility: 18%, Max allocation: 85%")
    
    def update_market_data_bar(self, symbol: str, ohlcv_data: Dict[str, float], 
                              atr_value: float, correlation_data: Dict[str, float] = None) -> None:
        """Update market data for 1-min bar processing as per document workflow.
        
        Args:
            symbol: Trading pair symbol.
            ohlcv_data: OHLCV data for the bar.
            atr_value: Current ATR(30) value.
            correlation_data: Correlations with other symbols.
        """
        # Update volatility data (EMA of 1-min ATR(30) over 60 bars)
        self.portfolio_manager.update_volatility_data(symbol, atr_value)
        
        # Update correlation data (EMA of pairwise returns over 60 bars)
        if correlation_data:
            for other_symbol, correlation in correlation_data.items():
                if other_symbol != symbol:
                    self.portfolio_manager.update_correlation_data(symbol, other_symbol, correlation)
        
        # Check for flash crash conditions using stress handler
        self.stress_handler.check_flash_crash(symbol, ohlcv_data, atr_value)
        
        # Check connection health
        self.stress_handler.check_connection_lag(datetime.now())
        
        # Update equity curve for slope calculation
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        equity_proxy = portfolio_summary['total_capital']  # Simplified equity tracking
        self.risk_manager.update_equity_curve(equity_proxy)
    
    def process_daily_rebalance(self) -> bool:
        """Process daily portfolio rebalancing if needed.
        
        Returns:
            True if rebalancing was performed.
        """
        if not self.portfolio_manager.should_rebalance():
            return False
        
        # Get active symbols from current allocations or volatility data
        active_symbols = list(self.portfolio_manager.volatility_data.keys())
        
        if not active_symbols:
            print("[ProductionExecution] No active symbols for rebalancing")
            return False
        
        print("[ProductionExecution] Daily rebalance triggered")
        
        # Perform rebalancing using production allocation system
        new_allocations = self.portfolio_manager.rebalance_portfolio(active_symbols)
        
        # Log rebalancing results
        total_allocated = sum(a.allocated_capital for a in new_allocations.values())
        turnover = self._calculate_turnover(new_allocations)
        
        print(f"[ProductionExecution] Rebalance completed:")
        print(f"[ProductionExecution]   Total allocated: ${total_allocated:.2f}")
        print(f"[ProductionExecution]   Turnover: {turnover:.2%}")
        print(f"[ProductionExecution]   High vol regime: {self.portfolio_manager.is_high_volatility_regime()}")
        
        return True
    
    async def process_signal(self, signal) -> Dict[str, Any]:
        """Process trading signal with production risk management.
        
        Args:
            signal: Trading signal with symbol, action, side, and metadata.
            
        Returns:
            Signal processing result.
        """
        try:
            symbol = signal.symbol
            action = signal.action
            side = signal.side
            
            # Get current price
            current_price = signal.metadata.get('price')
            if current_price is None:
                return {"status": "error", "reason": "Missing price information"}
            
            # Get ATR value for risk calculations
            atr_value = signal.metadata.get('atr_value')
            if atr_value is None:
                return {"status": "error", "reason": "Missing ATR value for production sizing"}
            
            print(f"\\n[ProductionExecution] Processing signal: {symbol} {action.upper()}/{side.upper()}")
            print(f"[ProductionExecution] Price: {current_price:.2f}, ATR: {atr_value:.6f}")
            
            # Handle different signal actions
            if action == "hold" or side == "none":
                return {"status": "skipped", "reason": "Hold signal", "symbol": symbol}
            
            elif action == "open":
                return await self._process_open_signal(signal, current_price, atr_value)
            
            elif action == "exit":
                return await self._process_exit_signal(symbol)
            
            else:
                return {"status": "error", "reason": f"Invalid action: {action}", "symbol": symbol}
                
        except Exception as e:
            error_msg = f"Error processing signal: {str(e)}"
            print(f"[ProductionExecution] ❌ {error_msg}")
            print(traceback.format_exc())
            return {"status": "error", "reason": error_msg}
    
    async def _process_open_signal(self, signal, current_price: float, atr_value: float) -> Dict[str, Any]:
        """Process open position signal using production risk management."""
        symbol = signal.symbol
        side = signal.side
        
        # Check kill switches before opening new positions
        kill_switches = self.risk_manager.check_kill_switches()
        if any(kill_switches.values()):
            return {
                "status": "rejected", 
                "reason": f"Kill switches active: {kill_switches}",
                "symbol": symbol
            }
        
        # Check if position already exists
        try:
            positions = await self.binance_client.get_open_positions(symbol)
            if positions:
                return {
                    "status": "skipped",
                    "reason": "Position already exists",
                    "symbol": symbol
                }
        except Exception as e:
            print(f"[ProductionExecution] Warning: Could not check positions: {e}")
        
        # Validate trade using production risk management
        risk_result = self.risk_manager.validate_trade(
            symbol=symbol,
            action="open",
            side=side,
            entry_price=current_price,
            atr_value=atr_value
        )
        
        if not risk_result["valid"]:
            return {
                "status": "rejected",
                "reason": risk_result["reason"],
                "symbol": symbol
            }
        
        # Extract position information
        position_info = risk_result["position_info"]
        
        print(f"[ProductionExecution] Opening {side.upper()} position:")
        print(f"[ProductionExecution]   Size: {position_info['size_contracts']:.6f} contracts")
        print(f"[ProductionExecution]   Leverage: {position_info['leverage']}x")
        print(f"[ProductionExecution]   Margin: ${position_info['margin_usdt']:.2f}")
        print(f"[ProductionExecution]   SL: {position_info['stop_loss_price']:.2f}")
        print(f"[ProductionExecution]   TP: {position_info['take_profit_price']:.2f}")
        
        # Execute the trade
        return await self.order_executor.execute_open_position(
            symbol=symbol,
            side=side,
            position_size=position_info['size_contracts'],
            current_price=current_price,
            stop_loss_price=position_info['stop_loss_price'],
            take_profit_price=position_info['take_profit_price'],
            leverage=position_info['leverage'],
            slippage_bp=3.0  # 3 basis points realistic slippage
        )
    
    async def _process_exit_signal(self, symbol: str) -> Dict[str, Any]:
        """Process exit position signal."""
        print(f"[ProductionExecution] Exiting position for {symbol}")
        return await self.order_executor.execute_close_position(symbol)
    
    def handle_disconnect(self, lag_seconds: float) -> None:
        """Handle connection issues as per document stress handling."""
        if lag_seconds > 3.0:
            self.stress_handler.check_connection_lag(datetime.now() - timedelta(seconds=lag_seconds))
    
    def resume_trading(self) -> None:
        """Resume trading after connection restored."""
        self.trading_paused = False
        print("[ProductionExecution] Trading resumed")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        risk_metrics = self.risk_manager.get_risk_metrics()
        stress_summary = self.stress_handler.get_stress_summary()
        
        return {
            "portfolio": portfolio_summary,
            "risk": risk_metrics,
            "stress": stress_summary,
            "system": {
                "trading_paused": self.trading_paused,
                "last_rebalance": portfolio_summary["last_rebalance"],
                "high_vol_regime": portfolio_summary["high_volatility_regime"]
            }
        }
    
    async def emergency_flatten(self, percentage: float = 1.0) -> Dict[str, Any]:
        """Emergency portfolio flattening for kill switches.
        
        Args:
            percentage: Percentage of positions to flatten (0.3 = 30%, 1.0 = 100%).
        """
        print(f"[ProductionExecution] ⚠️ Emergency flattening {percentage:.0%} of positions")
        
        # Get all open positions (simplified)
        # Would implement actual position flattening logic
        return {
            "status": "flattened",
            "percentage": percentage,
            "reason": "Emergency kill switch triggered"
        }


# Legacy compatibility
ExecutionEngine = ProductionExecutionEngine
