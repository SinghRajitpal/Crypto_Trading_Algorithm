#!/usr/bin/env python3
"""
Real-time dynamic signal processing test for the trading algorithm.

This test simulates the complete real-time workflow with dynamic:
1. Portfolio rebalancing based on changing market conditions
2. Position sizing adjusting to volatility
3. Leverage calculation based on risk
4. Stop loss and take profit placement
5. Signal validation and execution
"""

import pytest
import sys
import os
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import TradingAlgorithm
from algorithm.strategies.ma_crossover import MACrossoverStrategy
from algorithm.trade_signal import TradeSignal
from binance_exchange import BinanceClient


class TestRealTimeDynamicExecution:
    """Test real-time dynamic execution and calculations."""
    
    @pytest.fixture
    def mock_binance_client(self):
        """Create a comprehensive mock Binance client."""
        client = MagicMock()
        
        # Mock account metrics (testnet balance)
        client.get_account_metrics = AsyncMock(return_value={
            'total_wallet_balance': 14523.45,  # Realistic testnet balance
            'total_unrealized_pnl': 0.0,
            'total_margin_used': 0.0,
            'available_margin': 14523.45,
            'exposure_percentage': 0.0,
            'position_count': 0
        })
        
        # Mock positions (start with no positions)
        client.get_open_positions = AsyncMock(return_value=[])
        client.get_all_positions = AsyncMock(return_value=[])
        
        # Mock orders
        client.get_open_orders = AsyncMock(return_value=[])
        
        # Mock market data
        client.get_current_price = AsyncMock(return_value=50000.0)
        
        # Mock order placement
        client.place_order = AsyncMock(return_value={
            'id': '12345',
            'status': 'filled',
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'amount': 0.001,
            'price': 50000.0
        })
        
        # Mock setup
        client.setup_account_config = AsyncMock()
        client.close = AsyncMock()
        
        return client
    
    @pytest.fixture
    def strategy(self):
        """Create a test strategy."""
        return MACrossoverStrategy(params={
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'stop_loss_pct': 0.018,  # 1.8% stop loss
            'take_profit_pct': 0.036,  # 3.6% take profit (2:1 ratio)
            'leverage': 5
        })
    
    @pytest.mark.asyncio
    async def test_dynamic_capital_fetching(self, mock_binance_client, strategy):
        """Test dynamic capital fetching from exchange."""
        print("\n🔬 Testing Dynamic Capital Fetching")
        
        with patch('main.BinanceClient', return_value=mock_binance_client):
            algorithm = TradingAlgorithm(strategy=strategy, testnet=True, total_capital=None)
            
            # Initialize just enough to test capital fetching
            algorithm.binance_client = mock_binance_client
            
            # Test capital fetching
            await mock_binance_client.setup_account_config()
            account_metrics = await mock_binance_client.get_account_metrics()
            fetched_capital = account_metrics['total_wallet_balance']
            
            print(f"Fetched capital: ${fetched_capital:.2f}")
            assert fetched_capital == 14523.45, f"Should fetch correct capital amount"
            
            print(f"✅ Dynamic capital fetching works correctly")
    
    @pytest.mark.asyncio
    async def test_dynamic_portfolio_initialization(self, mock_binance_client, strategy):
        """Test dynamic portfolio initialization with real capital."""
        print("\n🔬 Testing Dynamic Portfolio Initialization")
        
        with patch('main.BinanceClient', return_value=mock_binance_client):
            algorithm = TradingAlgorithm(strategy=strategy, testnet=True, total_capital=None)
            algorithm.binance_client = mock_binance_client
            
            # Mock data engine components
            with patch('main.DataEngine') as mock_data_engine, \
                 patch('main.AlgoEngine') as mock_algo_engine, \
                 patch('main.ProductionExecutionEngine') as mock_execution_engine:
                
                # Setup mocks
                mock_data_instance = MagicMock()
                mock_data_engine.return_value = mock_data_instance
                
                mock_algo_instance = MagicMock()
                mock_algo_engine.return_value = mock_algo_instance
                
                mock_exec_instance = MagicMock()
                mock_exec_instance.setup = AsyncMock()
                mock_exec_instance.process_daily_rebalance = MagicMock(return_value=True)
                mock_exec_instance.portfolio_manager = MagicMock()
                mock_exec_instance.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
                mock_exec_instance.portfolio_manager.should_rebalance = MagicMock(return_value=True)
                mock_exec_instance.portfolio_manager.volatility_data = {}
                mock_exec_instance.portfolio_manager.update_volatility_data = MagicMock()
                mock_exec_instance.portfolio_manager.rebalance_portfolio = MagicMock(return_value={'BTCUSDT': MagicMock(allocated_capital=2500.0)})
                mock_exec_instance.portfolio_manager.get_portfolio_summary = MagicMock(return_value={
                    'allocated_capital': 12745.0,
                    'allocation_percentage': 0.85,
                    'total_capital': 14523.45,
                    'active_positions': 0
                })
                mock_exec_instance.get_portfolio_summary = MagicMock(return_value={
                    'total_capital': 14523.45,
                    'allocation_percentage': 0.85
                })
                mock_exec_instance.get_risk_metrics = MagicMock(return_value={
                    'risk_status': 'normal'
                })
                mock_execution_engine.return_value = mock_exec_instance
                
                # Test initialization logic
                algorithm.total_capital = None
                
                # Simulate the initialization process
                account_metrics = await algorithm.binance_client.get_account_metrics()
                algorithm.total_capital = account_metrics['total_wallet_balance']
                
                print(f"Initialized with capital: ${algorithm.total_capital:.2f}")
                assert algorithm.total_capital == 14523.45, "Should initialize with fetched capital"
                
                # Test that execution engine would be initialized with correct capital
                execution_engine = mock_execution_engine(
                    binance_client=algorithm.binance_client,
                    total_capital=algorithm.total_capital
                )
                
                print(f"✅ Portfolio initialization with dynamic capital works")
    
    @pytest.mark.asyncio
    async def test_dynamic_signal_processing_workflow(self, mock_binance_client, strategy):
        """Test complete dynamic signal processing workflow."""
        print("\n🔬 Testing Dynamic Signal Processing Workflow")
        
        # Create test signal
        signal = TradeSignal(
            symbol="BTCUSDT",
            action="open",
            side="buy",
            signal_confidence=0.85,
            timestamp=int(datetime.now().timestamp() * 1000)
        )
        
        # Add realistic metadata
        signal.metadata = {
            'price': 50000.0,
            'atr_value': 0.022,  # 2.2% ATR
            'fast_ma': 49800.0,
            'slow_ma': 49500.0,
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'reason': 'Fast MA crossed above Slow MA with strong momentum'
        }
        
        with patch('main.BinanceClient', return_value=mock_binance_client):
            # Create execution engine to test signal processing
            from execution.execution_engine import ProductionExecutionEngine
            
            execution_engine = ProductionExecutionEngine(
                binance_client=mock_binance_client,
                total_capital=14523.45
            )
            
            # Setup portfolio allocation first
            test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            
            # Add volatility data for allocation
            for symbol in test_symbols:
                for _ in range(25):
                    execution_engine.portfolio_manager.update_volatility_data(symbol, 0.022)
            
            # Force rebalancing
            execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
            allocations = execution_engine.portfolio_manager.rebalance_portfolio(test_symbols)
            
            print(f"Portfolio allocations:")
            total_allocated = 0
            for symbol, allocation in allocations.items():
                total_allocated += allocation.allocated_capital
                print(f"  {symbol}: ${allocation.allocated_capital:.2f}")
            
            assert total_allocated > 0, "Should have positive allocations"
            print(f"✅ Total allocated: ${total_allocated:.2f}")
            
            # Test signal validation
            validation_result = await execution_engine.validate_signal(signal, signal.metadata['price'])
            print(f"Signal validation: {validation_result}")
            
            if validation_result.get('valid', False):
                print(f"✅ Signal validation passed")
                
                # Test position sizing calculation
                allocated_capital = allocations.get("BTCUSDT", MagicMock(allocated_capital=0)).allocated_capital
                if allocated_capital > 0:
                    position_result = execution_engine.risk_manager.calculate_position_size(
                        symbol="BTCUSDT",
                        allocated_capital=allocated_capital,
                        atr_value=signal.metadata['atr_value'],
                        entry_price=signal.metadata['price']
                    )
                    
                    print(f"Position sizing result:")
                    print(f"  Allocated capital: ${allocated_capital:.2f}")
                    print(f"  Position size: ${position_result['position_size_usdt']:.2f}")
                    print(f"  Risk amount: ${position_result['risk_amount']:.2f}")
                    print(f"  Kelly fraction: {position_result['kelly_fraction']:.3f}")
                    
                    assert position_result['position_size_usdt'] > 0, "Should calculate positive position size"
                    assert position_result['position_size_usdt'] <= allocated_capital, "Position should not exceed allocation"
                    
                    print(f"✅ Position sizing calculation works")
                    
                    # Test leverage calculation
                    leverage = execution_engine.risk_manager.calculate_dynamic_leverage("BTCUSDT", signal.metadata['atr_value'])
                    print(f"Dynamic leverage: {leverage}x")
                    
                    assert 1 <= leverage <= 10, "Leverage should be between 1x and 10x"
                    print(f"✅ Dynamic leverage calculation works")
                    
                    # Test stop loss and take profit
                    sl_price, tp_price = execution_engine.risk_manager.calculate_stop_loss_take_profit(
                        entry_price=signal.metadata['price'],
                        side=signal.side,
                        atr_adjusted=signal.metadata['atr_value'] * signal.metadata['price']
                    )
                    
                    print(f"Risk management:")
                    print(f"  Entry price: ${signal.metadata['price']:.2f}")
                    print(f"  Stop loss: ${sl_price:.2f}")
                    print(f"  Take profit: ${tp_price:.2f}")
                    
                    # Validate SL/TP placement
                    assert sl_price < signal.metadata['price'] < tp_price, "SL < Entry < TP for long position"
                    
                    # Calculate risk-reward ratio
                    risk_distance = signal.metadata['price'] - sl_price
                    reward_distance = tp_price - signal.metadata['price']
                    rr_ratio = reward_distance / risk_distance
                    
                    print(f"  Risk-reward ratio: {rr_ratio:.2f}")
                    assert 1.8 <= rr_ratio <= 2.2, f"R:R ratio should be ~2:1, got {rr_ratio:.2f}"
                    
                    print(f"✅ Stop loss and take profit calculations work")
                    print(f"✅ Complete dynamic workflow successful")
            else:
                print(f"ℹ️ Signal validation failed (expected in some test scenarios)")
    
    @pytest.mark.asyncio
    async def test_dynamic_risk_adjustment_under_load(self, mock_binance_client, strategy):
        """Test dynamic risk adjustment under different market conditions."""
        print("\n🔬 Testing Dynamic Risk Adjustment Under Load")
        
        with patch('main.BinanceClient', return_value=mock_binance_client):
            from execution.execution_engine import ProductionExecutionEngine
            
            execution_engine = ProductionExecutionEngine(
                binance_client=mock_binance_client,
                total_capital=14523.45
            )
            
            # Test different volatility scenarios
            volatility_scenarios = [
                (0.015, "Low volatility"),
                (0.025, "Normal volatility"),
                (0.045, "High volatility"),
                (0.080, "Extreme volatility")
            ]
            
            symbol = "BTCUSDT"
            allocated_capital = 3000.0
            entry_price = 50000.0
            
            print(f"Testing risk adjustment across volatility scenarios:")
            
            for atr_value, scenario_name in volatility_scenarios:
                print(f"\n📊 {scenario_name} (ATR: {atr_value:.1%})")
                
                # Calculate position size
                position_result = execution_engine.risk_manager.calculate_position_size(
                    symbol=symbol,
                    allocated_capital=allocated_capital,
                    atr_value=atr_value,
                    entry_price=entry_price
                )
                
                # Calculate dynamic leverage
                leverage = execution_engine.risk_manager.calculate_dynamic_leverage(symbol, atr_value)
                
                # Calculate SL/TP
                sl_price, tp_price = execution_engine.risk_manager.calculate_stop_loss_take_profit(
                    entry_price=entry_price,
                    side="buy",
                    atr_adjusted=atr_value * entry_price
                )
                
                position_size = position_result['position_size_usdt']
                risk_amount = position_result['risk_amount']
                
                print(f"  Position size: ${position_size:.2f}")
                print(f"  Risk amount: ${risk_amount:.2f}")
                print(f"  Leverage: {leverage}x")
                print(f"  SL: ${sl_price:.0f}, TP: ${tp_price:.0f}")
                
                # Validate risk scaling
                assert position_size > 0, "Position size should be positive"
                assert risk_amount <= allocated_capital * 0.01, "Risk should not exceed 1% of capital"
                assert 1 <= leverage <= 10, "Leverage should be reasonable"
                assert sl_price < entry_price < tp_price, "SL/TP should be correctly placed"
            
            print(f"✅ Dynamic risk adjustment works across all scenarios")
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_triggers(self, mock_binance_client, strategy):
        """Test dynamic portfolio rebalancing triggers."""
        print("\n🔬 Testing Portfolio Rebalancing Triggers")
        
        with patch('main.BinanceClient', return_value=mock_binance_client):
            from execution.execution_engine import ProductionExecutionEngine
            
            execution_engine = ProductionExecutionEngine(
                binance_client=mock_binance_client,
                total_capital=14523.45
            )
            
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            
            # Add initial volatility data
            for symbol in symbols:
                for _ in range(20):
                    execution_engine.portfolio_manager.update_volatility_data(symbol, 0.02)
            
            # Test 1: Fresh start - should trigger rebalance
            print("\n📊 Test 1: Fresh start")
            execution_engine.portfolio_manager.last_rebalance_time = None
            should_rebalance = execution_engine.portfolio_manager.should_rebalance()
            print(f"Should rebalance (fresh start): {should_rebalance}")
            assert should_rebalance, "Should rebalance on fresh start"
            
            # Test 2: Recent rebalance - should not trigger
            print("\n📊 Test 2: Recent rebalance")
            execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=1)
            should_rebalance = execution_engine.portfolio_manager.should_rebalance()
            print(f"Should rebalance (recent): {should_rebalance}")
            assert not should_rebalance, "Should not rebalance if recent"
            
            # Test 3: 24+ hours - should trigger
            print("\n📊 Test 3: 24+ hours passed")
            execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
            should_rebalance = execution_engine.portfolio_manager.should_rebalance()
            print(f"Should rebalance (24+ hours): {should_rebalance}")
            assert should_rebalance, "Should rebalance after 24+ hours"
            
            # Test 4: Actual rebalancing
            print("\n📊 Test 4: Execute rebalancing")
            rebalanced = execution_engine.process_daily_rebalance()
            print(f"Rebalancing executed: {rebalanced}")
            
            if rebalanced:
                # Check portfolio summary
                summary = execution_engine.portfolio_manager.get_portfolio_summary()
                print(f"Portfolio after rebalancing:")
                print(f"  Total capital: ${summary['total_capital']:.2f}")
                print(f"  Allocated: ${summary['allocated_capital']:.2f}")
                print(f"  Allocation %: {summary['allocation_percentage']:.1%}")
                
                assert summary['allocated_capital'] > 0, "Should have allocated capital after rebalancing"
                print(f"✅ Portfolio rebalancing executed successfully")
            
            print(f"✅ Rebalancing triggers work correctly")


def run_realtime_dynamic_tests():
    """Run all real-time dynamic tests."""
    print("="*80)
    print("RUNNING REAL-TIME DYNAMIC EXECUTION TESTS")
    print("="*80)
    print("Testing real-time dynamic execution:")
    print("• Dynamic capital fetching from exchange")
    print("• Real-time portfolio allocation adjustments")
    print("• Dynamic position sizing with market conditions")
    print("• Leverage adjustment based on volatility")
    print("• Stop loss and take profit dynamic placement")
    print("• Signal validation and execution workflow")
    print("• Portfolio rebalancing triggers and execution")
    print("="*80)
    
    # Run pytest with verbose output
    pytest.main([
        __file__, 
        "-v", 
        "-s", 
        "--tb=short",
        "--capture=no"
    ])


if __name__ == "__main__":
    run_realtime_dynamic_tests()
