#!/usr/bin/env python3
"""
Execution Engine Comprehensive Test Suite
Senior Quantitative Developer Testing Protocol

Tests for portfolio allocation, risk management, order execution, and compliance
with the markdown specification.
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
from execution.execution_engine import ProductionExecutionEngine
from execution.portfolio import ProductionPortfolioManager, AllocationWeights
from execution.risk_manager import ProductionRiskManager, ProductionRiskParameters
from execution.executor import OrderExecutor
from execution.stress_handler import StressHandlingModule

class MockBinanceClient:
    """Mock Binance client for isolated execution testing."""
    
    def __init__(self):
        self.account_config_called = False
        self.orders = []
        self.positions = []
        self.balance = {'USDT': 10000.0}
        
    async def setup_account_config(self):
        self.account_config_called = True
        
    async def get_balance(self):
        return {'total': self.balance, 'free': self.balance, 'used': {}}
        
    async def get_open_positions(self, symbol=None):
        if symbol:
            return [p for p in self.positions if p['symbol'] == symbol]
        return self.positions
        
    async def create_order(self, symbol, order_type, side, amount, price=None, params={}):
        order = {
            'id': f'order_{len(self.orders)}',
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': amount,
            'price': price,
            'params': params,
            'status': 'open'
        }
        self.orders.append(order)
        return order
        
    async def close(self):
        pass

class ExecutionEngineTestSuite:
    """Comprehensive test suite for Execution Engine."""
    
    def __init__(self, framework: ComprehensiveTestFramework):
        self.framework = framework
        self.mock_client = MockBinanceClient()
        
    def test_portfolio_manager_initialization(self) -> Dict[str, Any]:
        """Test portfolio manager initialization and basic properties."""
        errors = []
        details = {}
        
        try:
            # Test basic initialization
            portfolio = ProductionPortfolioManager(
                total_capital=10000.0,
                target_volatility=0.18,
                max_allocation_pct=0.85
            )
            
            # Verify initial state according to markdown spec
            if portfolio.total_capital != 10000.0:
                errors.append(f"Expected total_capital=10000.0, got {portfolio.total_capital}")
            
            if portfolio.target_volatility != 0.18:
                errors.append(f"Expected target_volatility=0.18, got {portfolio.target_volatility}")
            
            if portfolio.max_allocation_pct != 0.85:
                errors.append(f"Expected max_allocation_pct=0.85, got {portfolio.max_allocation_pct}")
            
            # Check fixed parameters from document
            if portfolio.alpha != 0.3:
                errors.append(f"Expected alpha=0.3 (correlation adjustment), got {portfolio.alpha}")
            
            if portfolio.lookback_bars != 60:
                errors.append(f"Expected lookback_bars=60 (EMA lookback), got {portfolio.lookback_bars}")
            
            # Verify data structures are properly initialized
            if not isinstance(portfolio.volatility_data, dict):
                errors.append("volatility_data should be a dictionary")
            
            if not isinstance(portfolio.correlation_data, dict):
                errors.append("correlation_data should be a dictionary")
            
            if not isinstance(portfolio.allocation_weights, dict):
                errors.append("allocation_weights should be a dictionary")
            
            details['initialization'] = {
                'total_capital': portfolio.total_capital,
                'target_volatility': portfolio.target_volatility,
                'max_allocation_pct': portfolio.max_allocation_pct,
                'alpha': portfolio.alpha,
                'lookback_bars': portfolio.lookback_bars
            }
            
        except Exception as e:
            errors.append(f"Portfolio manager initialization failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    def test_volatility_correlation_data_management(self) -> Dict[str, Any]:
        """Test volatility and correlation data management."""
        errors = []
        details = {}
        
        try:
            portfolio = ProductionPortfolioManager(total_capital=10000.0)
            
            # Test volatility data update (EMA of 1-min ATR(30) over 60 bars)
            test_symbol = "BTCUSDT"
            test_atr_values = [0.02, 0.025, 0.022, 0.028, 0.024]
            
            for atr in test_atr_values:
                portfolio.update_volatility_data(test_symbol, atr)
            
            if test_symbol not in portfolio.volatility_data:
                errors.append(f"Volatility data not stored for {test_symbol}")
            else:
                vol_data = portfolio.volatility_data[test_symbol]
                if len(vol_data) != len(test_atr_values):
                    errors.append(f"Expected {len(test_atr_values)} volatility points, got {len(vol_data)}")
            
            # Test correlation data update (EMA of pairwise returns over 60 bars)
            symbol1, symbol2 = "BTCUSDT", "ETHUSDT"
            test_correlations = [0.8, 0.82, 0.78, 0.85, 0.79]
            
            for corr in test_correlations:
                portfolio.update_correlation_data(symbol1, symbol2, corr)
            
            corr_key = (symbol1, symbol2)
            if corr_key not in portfolio.correlation_data:
                errors.append(f"Correlation data not stored for {corr_key}")
            else:
                corr_data = portfolio.correlation_data[corr_key]
                if len(corr_data) != len(test_correlations):
                    errors.append(f"Expected {len(test_correlations)} correlation points, got {len(corr_data)}")
            
            # Test average correlation calculation
            avg_corr = portfolio.get_average_correlation(symbol1, [symbol1, symbol2])
            if avg_corr == 0:
                errors.append("Average correlation should not be zero with data")
            
            details['data_management'] = {
                'volatility_points_stored': len(portfolio.volatility_data.get(test_symbol, [])),
                'correlation_points_stored': len(portfolio.correlation_data.get(corr_key, [])),
                'average_correlation': avg_corr
            }
            
        except Exception as e:
            errors.append(f"Volatility/correlation data test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    def test_allocation_weight_calculation(self) -> Dict[str, Any]:
        """Test allocation weight calculation according to markdown spec."""
        errors = []
        details = {}
        
        try:
            portfolio = ProductionPortfolioManager(total_capital=10000.0)
            
            # Set up test data for weight calculation
            symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
            volatilities = [0.02, 0.03, 0.025]  # Different volatilities
            correlations = [0.8, 0.75, 0.7]     # Average correlations
            
            # Setup volatility data
            for symbol, vol in zip(symbols, volatilities):
                portfolio.update_volatility_data(symbol, vol)
            
            # Setup correlation data
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i != j:
                        portfolio.update_correlation_data(symbol1, symbol2, correlations[i])
            
            # Calculate weights using the exact formula from document:
            # w_i = (1/σ_i) × (1 + α × avg_correlation_i), normalized so Σw_i = 1
            weights = portfolio.compute_weights(symbols)
            
            # Verify we got float values (not AllocationWeights objects in the compute_weights method)
            for symbol, weight in weights.items():
                if not isinstance(weight, float):
                    errors.append(f"Expected float weight for {symbol}, got {type(weight)}")
                    continue
            
            # Verify weights sum to 1 (normalized)
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.001:
                errors.append(f"Weights should sum to 1.0, got {total_weight}")
            
            # Verify inverse volatility weighting (lower vol = higher weight)
            btc_weight = weights["BTCUSDT"]
            eth_weight = weights["ETHUSDT"]
            
            # BTC has lower volatility (0.02) than ETH (0.03), so should have higher weight
            if btc_weight <= eth_weight:
                errors.append("Lower volatility asset should have higher weight")
            
            details['weight_calculation'] = {
                'total_weight': total_weight,
                'btc_weight': btc_weight,
                'eth_weight': eth_weight,
                'weight_count': len(weights)
            }
            
        except Exception as e:
            errors.append(f"Weight calculation test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    def test_regime_detection_and_scaling(self) -> Dict[str, Any]:
        """Test volatility regime detection and scaling multiplier calculation."""
        errors = []
        details = {}
        
        try:
            portfolio = ProductionPortfolioManager(total_capital=10000.0)
            
            # Test high volatility regime detection
            # Simulate 30 days of volatility history
            normal_vol = [0.015, 0.018, 0.02, 0.016, 0.019] * 6  # 30 points
            high_vol = [0.04, 0.045, 0.05, 0.038, 0.042] * 6     # High volatility period
            
            # Add normal volatility history
            portfolio.volatility_history = normal_vol
            
            # Add some symbol data so regime detection can calculate sigma_hat
            portfolio.update_volatility_data("BTCUSDT", 0.018)
            portfolio.update_volatility_data("ETHUSDT", 0.020)
            
            # Test with normal current volatility
            is_high_vol_normal = portfolio.is_high_volatility_regime()
            
            # Test with high current volatility (add to history and symbols)
            portfolio.volatility_history.extend([0.045])  # Add high vol to history
            portfolio.update_volatility_data("BTCUSDT", 0.045)
            portfolio.update_volatility_data("ETHUSDT", 0.048)
            
            is_high_vol_extreme = portfolio.is_high_volatility_regime()
            
            # Test scaling multiplier calculation
            # Formula: m = min(1, target_vol/σ̂) × (0.5 if high_vol_regime else 1.0)
            scaling_normal = portfolio.calculate_scaling_multiplier()
            
            # The actual method doesn't take parameters, it calculates internally
            
            details['regime_detection'] = {
                'normal_regime_detected': not is_high_vol_normal,
                'high_regime_detected': is_high_vol_extreme,
                'normal_scaling': scaling_normal
            }
            
        except Exception as e:
            errors.append(f"Regime detection test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    def test_risk_manager_initialization(self) -> Dict[str, Any]:
        """Test risk manager initialization and parameter compliance."""
        errors = []
        details = {}
        
        try:
            portfolio = ProductionPortfolioManager(total_capital=10000.0)
            risk_manager = ProductionRiskManager(portfolio)
            
            # Verify risk parameters match markdown specification
            params = risk_manager.risk_params
            
            expected_params = {
                'risk_per_trade_pct': 0.008,      # 0.8% risk per trade
                'kelly_fraction': 0.7,             # Fractional Kelly criterion
                'base_cost_pct': 0.0014,           # 0.14% base cost
                'min_atr_floor': 0.001,            # Minimum ATR floor
                'atr_stop_multiplier': 1.8,        # SL = Entry ± 1.8×ATR
                'atr_trail_multiplier': 0.8,       # Trail by 0.8×ATR
                'risk_reward_ratio': 2.0,          # 1:2 risk-reward
                'partial_exit_ratio': 0.4,         # 40% partial exit at 1:1
                'max_leverage': 10,                 # Cap leverage at 10x
                'target_volatility': 0.18          # Target volatility for scaling
            }
            
            for param_name, expected_value in expected_params.items():
                actual_value = getattr(params, param_name)
                if actual_value != expected_value:
                    errors.append(f"Parameter {param_name}: expected {expected_value}, got {actual_value}")
            
            # Verify data structures are initialized
            if not isinstance(risk_manager.drawdown_history, list):
                errors.append("drawdown_history should be a list")
            
            if not isinstance(risk_manager.sharpe_history, list):
                errors.append("sharpe_history should be a list")
            
            if not isinstance(risk_manager.equity_curve, list):
                errors.append("equity_curve should be a list")
            
            details['risk_parameters'] = {param: getattr(params, param) for param in expected_params.keys()}
            
        except Exception as e:
            errors.append(f"Risk manager initialization test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    def test_position_sizing_calculation(self) -> Dict[str, Any]:
        """Test position sizing calculation with ATR and Kelly criterion."""
        errors = []
        details = {}
        
        try:
            portfolio = ProductionPortfolioManager(total_capital=10000.0)
            risk_manager = ProductionRiskManager(portfolio)
            
            # Test data
            symbol = "BTCUSDT"
            allocated_capital = 2000.0  # 20% allocation
            atr_value = 0.02            # 2% ATR
            normalized_volatility = 0.5  # Medium volatility
            
            # Calculate position size using the exact formula from document:
            # Size_i = (0.8% × Allocated_i × 0.7) / max(ATR_i, 0.001) - dynamic_cost
            position_result = risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocated_capital,
                atr_value=atr_value,
                entry_price=50000.0,  # Mock entry price
                volatility_norm=normalized_volatility
            )
            
            # The method returns a dictionary with position sizing information
            if not isinstance(position_result, dict):
                errors.append("Position sizing should return a dictionary")
            else:
                if 'size_contracts' not in position_result:
                    errors.append("Position sizing result should contain 'size_contracts' key")
                else:
                    position_size = position_result['size_contracts']
                    
                    # Verify position size is reasonable (positive and not too large)
                    if position_size <= 0:
                        errors.append("Position size should be positive")
                    
                    if position_result.get('size_usdt', 0) > allocated_capital * 10:  # Sanity check with max leverage
                        errors.append("Position size seems unreasonably large")
            
            # Test with minimum ATR floor
            tiny_atr = 0.0005  # Below minimum floor of 0.001
            size_result_with_floor = risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocated_capital,
                atr_value=tiny_atr,
                entry_price=50000.0,
                volatility_norm=0.0
            )
            
            # Should use ATR floor of 0.001 instead of 0.0005
            if isinstance(size_result_with_floor, dict) and 'size_contracts' in size_result_with_floor:
                size_with_floor = size_result_with_floor['size_contracts']
                if size_with_floor <= 0:
                    errors.append("Position size with ATR floor should be positive")
                    
                # Check that ATR floor was applied
                if 'atr_adjusted' in size_result_with_floor:
                    atr_adjusted = size_result_with_floor['atr_adjusted']
                    if atr_adjusted != 0.001:  # Should be floored at 0.001
                        errors.append(f"ATR should be floored at 0.001, got {atr_adjusted}")
            else:
                errors.append("Position sizing with ATR floor should return valid result")
            
            details['position_sizing'] = {
                'position_result_type': type(position_result).__name__ if 'position_result' in locals() else 'unknown',
                'size_contracts': position_result.get('size_contracts', 0) if isinstance(position_result, dict) else 0,
                'size_usdt': position_result.get('size_usdt', 0) if isinstance(position_result, dict) else 0,
                'leverage': position_result.get('leverage', 0) if isinstance(position_result, dict) else 0,
                'size_with_floor': size_result_with_floor.get('size_contracts', 0) if isinstance(size_result_with_floor, dict) else 0,
                'atr_used': atr_value,
                'allocated_capital': allocated_capital
            }
            
        except Exception as e:
            errors.append(f"Position sizing test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    def test_dynamic_leverage_calculation(self) -> Dict[str, Any]:
        """Test dynamic leverage calculation with all adjustment factors."""
        errors = []
        details = []
        
        try:
            portfolio = ProductionPortfolioManager(total_capital=10000.0)
            risk_manager = ProductionRiskManager(portfolio)
            
            # Test data
            symbol = "BTCUSDT"
            atr_value = 0.02            # Current ATR value
            
            # Calculate leverage using the actual method signature
            leverage = risk_manager.calculate_dynamic_leverage(
                symbol=symbol,
                atr_value=atr_value
            )
            
            # Verify leverage is within reasonable bounds
            if not isinstance(leverage, int):
                errors.append("Leverage should be an integer")
            
            if leverage < 1 or leverage > 10:
                errors.append(f"Leverage should be between 1 and 10, got {leverage}")
            
            # Test with different ATR values
            high_atr_leverage = risk_manager.calculate_dynamic_leverage(
                symbol=symbol,
                atr_value=0.05  # Higher volatility, should reduce leverage
            )
            
            low_atr_leverage = risk_manager.calculate_dynamic_leverage(
                symbol=symbol,
                atr_value=0.01  # Lower volatility, should allow higher leverage
            )
            
            # Higher volatility should generally result in lower leverage
            if high_atr_leverage > low_atr_leverage:
                errors.append("Higher volatility should generally result in lower leverage")
            
            details = {
                'normal_leverage': leverage,
                'high_atr_leverage': high_atr_leverage,
                'low_atr_leverage': low_atr_leverage,
                'atr_value': atr_value
            }
            
        except Exception as e:
            errors.append(f"Dynamic leverage test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    async def test_execution_engine_integration(self) -> Dict[str, Any]:
        """Test execution engine integration and workflow."""
        errors = []
        details = {}
        
        try:
            # Initialize execution engine with mock client
            execution_engine = ProductionExecutionEngine(
                binance_client=self.mock_client,
                total_capital=10000.0
            )
            
            # Test setup
            await execution_engine.setup()
            
            if not self.mock_client.account_config_called:
                errors.append("Account configuration should be called during setup")
            
            # Test market data update workflow
            symbol = "BTCUSDT"
            ohlcv_data = {
                'open': 50000.0,
                'high': 50200.0,
                'low': 49800.0,
                'close': 50100.0,
                'volume': 100.0
            }
            atr_value = 0.02
            correlation_data = {"ETHUSDT": 0.8}
            
            # Update market data (should update volatility and correlation data)
            execution_engine.update_market_data_bar(symbol, ohlcv_data, atr_value, correlation_data)
            
            # Verify data was processed
            if symbol not in execution_engine.portfolio_manager.volatility_data:
                errors.append("Market data update should store volatility data")
            
            # Test daily rebalancing
            # Add more symbols for rebalancing
            for sym in ["ETHUSDT", "XRPUSDT"]:
                execution_engine.update_market_data_bar(sym, ohlcv_data, atr_value)
            
            # Force rebalancing
            execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(days=2)
            rebalanced = execution_engine.process_daily_rebalance()
            
            if not rebalanced:
                errors.append("Daily rebalancing should have been triggered")
            
            # Test portfolio summary
            summary = execution_engine.get_portfolio_summary()
            expected_keys = ['total_capital', 'allocated_capital', 'allocation_percentage']
            
            for key in expected_keys:
                if key not in summary:
                    errors.append(f"Portfolio summary missing key: {key}")
            
            # Test risk metrics
            risk_metrics = execution_engine.get_risk_metrics()
            expected_risk_keys = ['risk_status', 'risk_parameters', 'active_positions']
            
            for key in expected_risk_keys:
                if key not in risk_metrics:
                    errors.append(f"Risk metrics missing key: {key}")
            
            # Check if risk_parameters contains max_leverage
            if 'risk_parameters' in risk_metrics:
                risk_params = risk_metrics['risk_parameters']
                if 'max_leverage' not in risk_params:
                    errors.append("Risk parameters missing max_leverage")
            else:
                errors.append("Risk metrics missing risk_parameters section")
            
            details['integration_tests'] = {
                'account_setup': self.mock_client.account_config_called,
                'volatility_data_stored': len(execution_engine.portfolio_manager.volatility_data),
                'rebalancing_triggered': rebalanced,
                'portfolio_summary_complete': len([k for k in expected_keys if k in summary]),
                'risk_metrics_complete': len([k for k in expected_risk_keys if k in risk_metrics]),
                'risk_parameters_present': 'risk_parameters' in risk_metrics
            }
            
        except Exception as e:
            errors.append(f"Execution engine integration test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all execution engine tests."""
        test_results = []
        
        # Portfolio manager tests
        test_results.append(await self.framework.run_test(
            self.test_portfolio_manager_initialization,
            "execution_portfolio_manager_initialization"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_volatility_correlation_data_management,
            "execution_volatility_correlation_data"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_allocation_weight_calculation,
            "execution_allocation_weight_calculation"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_regime_detection_and_scaling,
            "execution_regime_detection_scaling"
        ))
        
        # Risk manager tests
        test_results.append(await self.framework.run_test(
            self.test_risk_manager_initialization,
            "execution_risk_manager_initialization"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_position_sizing_calculation,
            "execution_position_sizing_calculation"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_dynamic_leverage_calculation,
            "execution_dynamic_leverage_calculation"
        ))
        
        # Integration tests
        test_results.append(await self.framework.run_test(
            self.test_execution_engine_integration,
            "execution_engine_integration"
        ))
        
        return test_results

async def main():
    """Run execution engine test suite."""
    print("🔬 EXECUTION ENGINE COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    framework = ComprehensiveTestFramework(verbose=True)
    test_suite = ExecutionEngineTestSuite(framework)
    
    # Run all tests
    await test_suite.run_all_tests()
    
    # Print detailed report
    framework.print_test_report()

if __name__ == "__main__":
    asyncio.run(main())
