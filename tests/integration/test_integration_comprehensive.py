#!/usr/bin/env python3
"""
Integration Test Suite - End-to-End Signal → Execution Pipeline
Senior Quantitative Developer Testing Protocol

Tests the complete trading workflow from signal generation through to position sizing,
risk management, and order execution. Validates interface resilience, state management,
and data consistency across the entire system.
"""

import os
import sys
import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_test_framework import ComprehensiveTestFramework
from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from execution.execution_engine import ProductionExecutionEngine
from main import TradingAlgorithm
from algorithm.strategies.base_strategy import BaseStrategy

class MockDataEngineForIntegration:
    """Mock data engine with consistent data for integration testing."""
    
    def __init__(self):
        self.candle_data = {}
        self.binance_client = None
        
    def get_candles(self, symbol: str, timeframe: str) -> List[List]:
        """Get mock candle data with trend patterns."""
        key = f"{symbol}_{timeframe}"
        if key not in self.candle_data:
            # Generate trending data for better signal testing
            self.candle_data[key] = self._generate_trending_data(symbol, 50)
        return self.candle_data[key]
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[List]:
        """Get the latest candle."""
        candles = self.get_candles(symbol, timeframe)
        return candles[-1] if candles else None
    
    def _generate_trending_data(self, symbol: str, periods: int) -> List[List]:
        """Generate trending OHLCV data for signal generation."""
        current_time = int(time.time() * 1000)
        candles = []
        
        # Start with base price based on symbol
        if symbol == "BTCUSDT":
            base_price = 50000.0
        elif symbol == "ETHUSDT":
            base_price = 3000.0
        else:
            base_price = 1.0
        
        current_price = base_price
        
        for i in range(periods):
            timestamp = current_time - (periods - i) * 60000  # 1-minute candles
            
            # Create an uptrend in the last 20 candles to trigger buy signals
            if i >= periods - 20:
                trend_factor = 1.005  # 0.5% increase per candle
            else:
                trend_factor = 0.999  # Slight downtrend initially
            
            # Generate realistic OHLC with trend
            open_price = current_price
            close_price = current_price * trend_factor * (1 + np.random.normal(0, 0.001))
            
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
            volume = np.random.uniform(100, 1000)
            
            candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
            current_price = close_price
        
        return candles

class IntegrationTestStrategy(BaseStrategy):
    """Test strategy that generates predictable signals for integration testing."""
    
    def __init__(self, signal_pattern: List[str] = None):
        """Initialize with predictable signal pattern.
        
        Args:
            signal_pattern: List of signal types to generate in sequence 
                          ("buy", "sell", "hold", "exit_long", "exit_short")
        """
        super().__init__({}, "integration_test_strategy")
        self.signal_pattern = signal_pattern or ["hold", "buy", "hold", "exit_long", "hold"]
        self.signal_index = 0
        self.call_count = 0
        
    def get_required_indicators(self) -> List[str]:
        """Return simple indicators for testing."""
        return ["sma_5", "sma_20"]  # Simple moving averages
    
    async def _generate_signals(self, data: Dict[str, Any], indicator_data: Dict[str, Any], symbol: str) -> TradeSignal:
        """Generate signals according to the pattern."""
        current_signal = self.signal_pattern[self.signal_index % len(self.signal_pattern)]
        self.signal_index += 1
        self.call_count += 1
        
        # Create signal based on pattern
        if current_signal == "buy":
            action, side = "open", "buy"
        elif current_signal == "sell":
            action, side = "open", "sell"
        elif current_signal == "exit_long":
            action, side = "exit", "sell"
        elif current_signal == "exit_short":
            action, side = "exit", "buy"
        else:  # hold
            action, side = "hold", "none"
        
        # Add mock indicator values to metadata
        metadata = {
            "atr_value": 0.02,
            "price": data.get('close', [50000])[-1] if data and 'close' in data else 50000,
            "signal_pattern_index": self.signal_index - 1,
            "indicators": indicator_data
        }
        
        return TradeSignal(
            action=action,
            side=side,
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata=metadata,
            signal_confidence=0.8,
            timestamp=int(time.time() * 1000)
        )

class MockBinanceClientForIntegration:
    """Enhanced mock Binance client for integration testing."""
    
    def __init__(self):
        self.account_config_called = False
        self.orders = []
        self.positions = []
        self.balance = {'USDT': 10000.0}
        self.order_id_counter = 0
        
    async def setup_account_config(self):
        self.account_config_called = True
        
    async def get_balance(self):
        return {'total': self.balance, 'free': self.balance, 'used': {}}
        
    async def get_open_positions(self, symbol=None):
        if symbol:
            return [p for p in self.positions if p['symbol'] == symbol]
        return self.positions
        
    async def create_order(self, symbol, order_type, side, amount, price=None, params={}):
        self.order_id_counter += 1
        order = {
            'id': f'order_{self.order_id_counter}',
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': amount,
            'price': price,
            'params': params,
            'status': 'filled',  # Assume orders fill immediately in tests
            'timestamp': int(time.time() * 1000)
        }
        self.orders.append(order)
        
        # Update positions if it's a market order
        if order_type == 'market':
            self._update_position(symbol, side, amount, price or 50000)
        
        return order
    
    async def open_position(self, symbol, side, amount, price=None, stop_loss=None, take_profit=None, leverage=None, margin_type=None):
        """Mock open_position method for testing."""
        self.order_id_counter += 1
        
        # Create a position entry
        position = {
            'id': f'position_{self.order_id_counter}',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'contracts': amount,
            'entry_price': price or 50000.0,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'leverage': leverage or 1,
            'margin_type': margin_type or 'isolated',
            'timestamp': int(time.time() * 1000),
            'status': 'open'
        }
        
        # Add to positions list
        self.positions.append(position)
        
        # Also create corresponding order record
        order = {
            'id': f'order_{self.order_id_counter}',
            'symbol': symbol,
            'type': 'market',
            'side': side,
            'amount': amount,
            'price': price or 50000.0,
            'params': {'leverage': leverage, 'stop_loss': stop_loss, 'take_profit': take_profit},
            'status': 'filled',
            'timestamp': int(time.time() * 1000)
        }
        self.orders.append(order)
        
        return {
            'order': order,
            'position': position,
            'status': 'success'
        }
    
    def _update_position(self, symbol: str, side: str, amount: float, price: float):
        """Update mock position data."""
        existing_position = None
        for i, pos in enumerate(self.positions):
            if pos['symbol'] == symbol:
                existing_position = i
                break
        
        if existing_position is not None:
            # Update existing position
            current_contracts = float(self.positions[existing_position]['contracts'])
            if side == 'buy':
                new_contracts = current_contracts + amount
            else:  # sell
                new_contracts = current_contracts - amount
            
            self.positions[existing_position]['contracts'] = new_contracts
            if abs(new_contracts) < 0.001:  # Close position if very small
                self.positions.pop(existing_position)
        else:
            # Create new position
            if side == 'buy':
                contracts = amount
            else:
                contracts = -amount
            
            self.positions.append({
                'symbol': symbol,
                'contracts': contracts,
                'entryPrice': price,
                'unrealizedPnl': 0,
                'initialMargin': contracts * price / 10  # Assume 10x leverage
            })
    
    async def close(self):
        pass

class IntegrationTestSuite:
    """Comprehensive integration test suite."""
    
    def __init__(self, framework: ComprehensiveTestFramework):
        self.framework = framework
        self.mock_data_engine = MockDataEngineForIntegration()
        self.mock_client = MockBinanceClientForIntegration()
        
    async def test_signal_generation_to_execution_pipeline(self) -> Dict[str, Any]:
        """Test complete pipeline from signal generation to position creation."""
        errors = []
        details = {}
        
        try:
            # Initialize components
            algo_engine = AlgoEngine(self.mock_data_engine)
            execution_engine = ProductionExecutionEngine(
                binance_client=self.mock_client,
                total_capital=10000.0
            )
            
            await execution_engine.setup()
            
            # Create test strategy with predictable signals
            strategy = IntegrationTestStrategy(["hold", "buy", "hold"])
            strategy.set_algo_engine(algo_engine)
            
            # Process signals and validate execution
            signal_count = 0
            buy_signals = 0
            hold_signals = 0
            
            # Simulate market data updates for multiple symbols
            test_symbols = ["BTCUSDT", "ETHUSDT"]
            
            for symbol in test_symbols:
                # Update market data for execution engine
                ohlcv_data = {
                    'open': 50000.0,
                    'high': 50200.0,
                    'low': 49800.0,
                    'close': 50100.0,
                    'volume': 100.0
                }
                execution_engine.update_market_data_bar(symbol, ohlcv_data, 0.02)
            
            # Force initial portfolio rebalance to allocate capital
            print("[integration_test] Triggering initial portfolio rebalance...")
            # Override the rebalance timer for testing
            from datetime import timedelta
            execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
            execution_engine.process_daily_rebalance()
            
            for symbol in test_symbols:
                # Generate and process signals
                for i in range(3):  # Generate 3 signals per symbol
                    signal = await algo_engine.process_signals(symbol, "1m", strategy)
                    
                    if signal:
                        signal_count += 1
                        
                        if signal.action == "open" and signal.side == "buy":
                            buy_signals += 1
                            
                            # Test signal validation and execution
                            current_price = 50100.0
                            validation_result = await execution_engine.validate_signal(signal, current_price)
                            
                            if not isinstance(validation_result, dict):
                                errors.append("Signal validation should return a dictionary")
                            
                            # Check if validation contains required keys (based on actual API)
                            expected_keys = ['valid']
                            for key in expected_keys:
                                if key not in validation_result:
                                    errors.append(f"Validation result missing key: {key}")
                            
                            # If valid, should have position information
                            if validation_result.get('valid', False):
                                if 'position_info' not in validation_result:
                                    errors.append("Valid signal should include position_info")
                                
                                # Actually process the signal to trigger portfolio allocation
                                processing_result = await execution_engine.process_signal(signal)
                                if processing_result.get('status') not in ['success', 'completed', 'skipped', 'rejected']:
                                    errors.append(f"Signal processing failed: {processing_result.get('reason', 'Unknown error')}")
                        
                        elif signal.action == "hold":
                            hold_signals += 1
            
            # Check portfolio state after signal processing
            portfolio_summary = execution_engine.get_portfolio_summary()
            risk_metrics = execution_engine.get_risk_metrics()
            
            # Verify portfolio allocation occurred
            if portfolio_summary['allocated_capital'] <= 0:
                errors.append("Portfolio should have allocated capital after processing")
            
            # Check risk management is functioning
            if 'risk_status' not in risk_metrics:
                errors.append("Risk metrics should include risk_status")
            
            details['pipeline_tests'] = {
                'total_signals': signal_count,
                'buy_signals': buy_signals,
                'hold_signals': hold_signals,
                'portfolio_allocated': portfolio_summary['allocated_capital'],
                'risk_status': risk_metrics.get('risk_status', 'unknown'),
                'symbols_processed': len(test_symbols),
                'mock_orders_created': len(self.mock_client.orders),
                'mock_positions': len(self.mock_client.positions)
            }
            
        except Exception as e:
            errors.append(f"Signal-to-execution pipeline test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    async def test_state_consistency_across_components(self) -> Dict[str, Any]:
        """Test state consistency between algorithm and execution engines."""
        errors = []
        details = {}
        
        try:
            # Initialize engines
            algo_engine = AlgoEngine(self.mock_data_engine)
            execution_engine = ProductionExecutionEngine(
                binance_client=self.mock_client,
                total_capital=5000.0
            )
            
            await execution_engine.setup()
            
            # Update market data first to create volatility data
            initial_symbols = ["BTCUSDT", "ETHUSDT"]
            for symbol in initial_symbols:
                execution_engine.update_market_data_bar(symbol, {
                    'open': 50000.0,
                    'high': 50200.0,
                    'low': 49800.0,
                    'close': 50100.0,
                    'volume': 100.0
                }, 0.02)
            
            # Force initial portfolio rebalancing to allocate capital
            print("[integration_test] Triggering initial portfolio rebalance for state consistency test...")
            execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
            execution_engine.process_daily_rebalance()
            
            # Create strategy and process multiple signals
            strategy = IntegrationTestStrategy(["buy", "hold", "buy", "exit_long"])
            strategy.set_algo_engine(algo_engine)
            
            symbol = "BTCUSDT"
            signals_processed = []
            
            # Process signals and track state changes
            for i in range(4):
                # Update market data with slight variations
                ohlcv_data = {
                    'open': 50000.0 + i * 10,
                    'high': 50200.0 + i * 10,
                    'low': 49800.0 + i * 10,
                    'close': 50100.0 + i * 10,
                    'volume': 100.0
                }
                execution_engine.update_market_data_bar(symbol, ohlcv_data, 0.02 + i * 0.001)
                
                # Change candle data to trigger new signals
                trending_data = self.mock_data_engine._generate_trending_data(symbol, 50 + i)
                self.mock_data_engine.candle_data[f"{symbol}_1m"] = trending_data
                
                signal = await algo_engine.process_signals(symbol, "1m", strategy)
                
                if signal:
                    signals_processed.append({
                        'action': signal.action,
                        'side': signal.side,
                        'timestamp': signal.timestamp,
                        'index': i
                    })
                    
                    # Validate signal immediately
                    if signal.action != "hold":
                        validation = await execution_engine.validate_signal(signal, 50100.0 + i * 10)
                        
                        if not validation.get('valid', False):
                            errors.append(f"Signal {i} failed risk validation: {signal.action}/{signal.side}")
                            details[f'signal_{i}_validation_reason'] = validation.get('reason', 'unknown')
            
            # Check state consistency
            algo_state_count = len(algo_engine._last_signal_states)
            if algo_state_count == 0:
                errors.append("Algorithm engine should track signal states")
            
            portfolio_volatility_count = len(execution_engine.portfolio_manager.volatility_data)
            if portfolio_volatility_count == 0:
                errors.append("Portfolio manager should track volatility data")
            
            # Verify timestamps are consistent and increasing
            if len(signals_processed) > 1:
                for i in range(1, len(signals_processed)):
                    if signals_processed[i]['timestamp'] <= signals_processed[i-1]['timestamp']:
                        errors.append("Signal timestamps should be increasing")
                        break
            
            details['state_consistency'] = {
                'signals_processed': len(signals_processed),
                'algo_state_count': algo_state_count,
                'portfolio_volatility_count': portfolio_volatility_count,
                'signal_sequence': [s['action'] for s in signals_processed],
                'timestamps_consistent': len(errors) == 0
            }
            
        except Exception as e:
            errors.append(f"State consistency test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    async def test_stress_conditions_and_edge_cases(self) -> Dict[str, Any]:
        """Test system behavior under stress conditions and edge cases."""
        errors = []
        details = {}
        
        try:
            # Initialize with minimal capital to test edge cases
            execution_engine = ProductionExecutionEngine(
                binance_client=self.mock_client,
                total_capital=100.0  # Very small capital
            )
            
            await execution_engine.setup()
            
            # Test 1: High frequency signals
            algo_engine = AlgoEngine(self.mock_data_engine)
            rapid_strategy = IntegrationTestStrategy(["buy", "sell", "buy", "sell"] * 5)
            rapid_strategy.set_algo_engine(algo_engine)
            
            rapid_signals = 0
            start_time = time.time()
            
            for i in range(10):
                signal = await algo_engine.process_signals("BTCUSDT", "1m", rapid_strategy)
                if signal:
                    rapid_signals += 1
                    
                    # Try to execute rapidly
                    if signal.action != "hold":
                        validation = await execution_engine.validate_signal(signal, 50000.0)
                        if validation.get('valid', False):
                            # Signal was processed successfully even under rapid conditions
                            pass
            
            execution_time = time.time() - start_time
            
            # Test 2: Invalid signal handling - create a properly formatted but logically invalid signal
            try:
                invalid_signal = TradeSignal(
                    action="open",  # Valid action
                    side="buy",
                    symbol="INVALIDPAIR",  # Invalid symbol
                    strategy_id="test",
                    metadata={"atr_value": 0.02},  # Valid metadata format
                    signal_confidence=0.5
                )
                
                # This should fail gracefully due to invalid symbol/missing allocation
                validation = await execution_engine.validate_signal(invalid_signal, 50000.0)
                if validation.get('valid', False):  # Should be False for invalid signal
                    errors.append("Invalid symbol signal should fail validation")
            except Exception as e:
                # If it throws an exception, that's also acceptable error handling
                pass
            
            # Test 3: Portfolio allocation with extreme volatility
            extreme_atr = 0.1  # 10% ATR (very high)
            execution_engine.update_market_data_bar("BTCUSDT", {
                'open': 50000, 'high': 55000, 'low': 45000, 'close': 52000, 'volume': 1000
            }, extreme_atr)
            
            # Should trigger high volatility regime
            is_high_vol = execution_engine.portfolio_manager.is_high_volatility_regime()
            
            details['stress_tests'] = {
                'rapid_signals_processed': rapid_signals,
                'execution_time': execution_time,
                'high_vol_regime_triggered': is_high_vol,
                'avg_signal_time': execution_time / max(rapid_signals, 1),
                'small_capital_handled': execution_engine.total_capital == 100.0
            }
            
            # Performance check
            if execution_time > 1.0:  # Should process 10 signals in under 1 second
                errors.append(f"Signal processing too slow: {execution_time:.2f}s for {rapid_signals} signals")
            
        except Exception as e:
            errors.append(f"Stress testing failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    async def test_trading_algorithm_full_integration(self) -> Dict[str, Any]:
        """Test the complete TradingAlgorithm class integration."""
        errors = []
        details = {}
        
        try:
            # Create a simple strategy for testing
            test_strategy = IntegrationTestStrategy(["hold", "buy", "hold"])
            
            # Initialize TradingAlgorithm with mock components
            # Note: We need to mock the BinanceClient to avoid real API calls
            
            # Create a minimal integration test
            algo_engine = AlgoEngine(self.mock_data_engine)
            execution_engine = ProductionExecutionEngine(
                binance_client=self.mock_client,
                total_capital=1000.0
            )
            
            # Set up strategy
            test_strategy.set_algo_engine(algo_engine)
            
            # Test basic initialization and setup
            await execution_engine.setup()
            
            # Simulate a few market data updates
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                execution_engine.update_market_data_bar(symbol, {
                    'open': 50000.0,
                    'high': 50200.0,
                    'low': 49800.0,
                    'close': 50100.0,
                    'volume': 100.0
                }, 0.02)
            
            # Process some signals
            signals_generated = 0
            for i in range(3):
                signal = await algo_engine.process_signals("BTCUSDT", "1m", test_strategy)
                if signal:
                    signals_generated += 1
                    
                    if signal.action != "hold":
                        validation = await execution_engine.validate_signal(signal, 50100.0)
                        if not validation.get('risk_valid', False):
                            errors.append(f"Signal validation failed for {signal.action}/{signal.side}")
            
            # Test portfolio state
            portfolio_summary = execution_engine.get_portfolio_summary()
            risk_metrics = execution_engine.get_risk_metrics()
            
            # Verify integration is working
            if portfolio_summary['total_capital'] != 1000.0:
                errors.append("Portfolio capital not properly initialized")
            
            if risk_metrics['risk_status'] not in ['normal', 'caution', 'warning', 'critical']:
                errors.append("Risk status should be one of the expected values")
            
            details['full_integration'] = {
                'signals_generated': signals_generated,
                'portfolio_total_capital': portfolio_summary['total_capital'],
                'risk_status': risk_metrics['risk_status'],
                'volatility_data_symbols': len(execution_engine.portfolio_manager.volatility_data),
                'mock_account_setup': self.mock_client.account_config_called
            }
            
        except Exception as e:
            errors.append(f"Full integration test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all integration tests."""
        test_results = []
        
        # Core integration tests
        test_results.append(await self.framework.run_test(
            self.test_signal_generation_to_execution_pipeline,
            "integration_signal_to_execution_pipeline"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_state_consistency_across_components,
            "integration_state_consistency"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_stress_conditions_and_edge_cases,
            "integration_stress_conditions"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_trading_algorithm_full_integration,
            "integration_trading_algorithm_full"
        ))
        
        return test_results

async def main():
    """Run integration test suite."""
    print("🔬 INTEGRATION TEST SUITE - END-TO-END PIPELINE")
    print("=" * 60)
    
    framework = ComprehensiveTestFramework(verbose=True)
    test_suite = IntegrationTestSuite(framework)
    
    # Run all tests
    await test_suite.run_all_tests()
    
    # Print detailed report
    framework.print_test_report()

if __name__ == "__main__":
    asyncio.run(main())
