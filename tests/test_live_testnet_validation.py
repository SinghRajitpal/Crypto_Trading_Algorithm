#!/usr/bin/env python3
"""
Live Testnet Validation Tests for Crypto Trading Algorithm.

This module provides comprehensive testing on Binance testnet to validate
real-world behavior of the trading system.

Tests cover:
- Account connectivity and setup
- Order placement and management
- SL/TP order pairs
- Position tracking
- Real-time data processing
- Complete trading workflow validation
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_exchange import BinanceClient
from execution.execution_engine import ProductionExecutionEngine
from execution.portfolio import ProductionPortfolioManager
from execution.risk_manager import ProductionRiskManager
from algorithm.trade_signal import TradeSignal
import config

# Set up logging for testnet validation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.live
@pytest.mark.asyncio
class TestLiveTestnetValidation:
    """Live testnet validation test suite."""
    
    @pytest.fixture(scope="class")
    async def binance_client(self):
        """Set up live testnet Binance client."""
        client = BinanceClient(testnet=True)
        try:
            await client.setup_account_config()
            yield client
        finally:
            await client.close()
    
    @pytest.fixture(scope="class")
    async def execution_engine(self, binance_client):
        """Set up production execution engine with testnet client."""
        engine = ProductionExecutionEngine(binance_client, total_capital=5000.0)
        await engine.setup()
        return engine
    
    async def test_testnet_connectivity(self, binance_client):
        """Test basic testnet connectivity and account access."""
        logger.info("Testing testnet connectivity...")
        
        # Test account info access
        try:
            account_info = await binance_client.get_account_balance()
            assert account_info is not None, "Should receive account information"
            assert "USDT" in account_info, "Should have USDT balance information"
            
            usdt_balance = float(account_info["USDT"]["free"])
            logger.info(f"USDT balance: ${usdt_balance:.2f}")
            
            # Verify sufficient balance for testing
            assert usdt_balance >= 100.0, f"Insufficient USDT balance for testing: ${usdt_balance:.2f}"
            
        except Exception as e:
            pytest.fail(f"Failed to access testnet account: {e}")
    
    async def test_market_data_access(self, binance_client):
        """Test real-time market data access."""
        logger.info("Testing market data access...")
        
        test_symbol = "BTCUSDT"
        
        try:
            # Test ticker data
            ticker = await binance_client.get_ticker(test_symbol)
            assert ticker is not None, "Should receive ticker data"
            assert "price" in ticker, "Ticker should contain price"
            
            price = float(ticker["price"])
            assert price > 0, "Price should be positive"
            logger.info(f"{test_symbol} price: ${price:.2f}")
            
            # Test orderbook data
            orderbook = await binance_client.get_orderbook(test_symbol, limit=10)
            assert orderbook is not None, "Should receive orderbook data"
            assert "bids" in orderbook and "asks" in orderbook, "Orderbook should have bids and asks"
            
            # Verify orderbook structure
            assert len(orderbook["bids"]) > 0, "Should have bid orders"
            assert len(orderbook["asks"]) > 0, "Should have ask orders"
            
            # Test spread calculation
            best_bid = float(orderbook["bids"][0][0])
            best_ask = float(orderbook["asks"][0][0])
            spread = (best_ask - best_bid) / best_bid
            
            logger.info(f"{test_symbol} spread: {spread:.4%}")
            assert spread >= 0, "Spread should be non-negative"
            assert spread <= 0.01, "Spread should be reasonable for major pair"
            
        except Exception as e:
            pytest.fail(f"Failed to access market data: {e}")
    
    async def test_position_management(self, binance_client):
        """Test position information and management."""
        logger.info("Testing position management...")
        
        try:
            # Get current positions
            positions = await binance_client.get_position_info()
            assert isinstance(positions, list), "Positions should be a list"
            
            logger.info(f"Current positions: {len(positions)}")
            
            # Check position structure for any existing positions
            for position in positions:
                if float(position.get("positionAmt", 0)) != 0:
                    logger.info(f"Open position: {position['symbol']} - {position['positionAmt']}")
                    
                    # Verify position structure
                    required_fields = ["symbol", "positionAmt", "entryPrice", "percentage", "unrealizedPnl"]
                    for field in required_fields:
                        assert field in position, f"Position should have {field} field"
            
        except Exception as e:
            pytest.fail(f"Failed to access position information: {e}")
    
    async def test_order_validation(self, binance_client):
        """Test order validation without actually placing orders."""
        logger.info("Testing order validation...")
        
        test_symbol = "BTCUSDT"
        
        try:
            # Get current market price
            ticker = await binance_client.get_ticker(test_symbol)
            market_price = float(ticker["price"])
            
            # Test order parameters for a small buy order
            test_quantity = 0.001  # Very small quantity for testing
            test_price = market_price * 0.95  # 5% below market (limit order)
            
            # Validate order parameters (without placing)
            order_params = {
                "symbol": test_symbol,
                "side": "BUY",
                "type": "LIMIT",
                "quantity": test_quantity,
                "price": test_price,
                "timeInForce": "GTC"
            }
            
            # Test parameter validation
            assert order_params["quantity"] > 0, "Quantity should be positive"
            assert order_params["price"] > 0, "Price should be positive"
            assert order_params["side"] in ["BUY", "SELL"], "Side should be BUY or SELL"
            
            logger.info(f"Order validation passed: {order_params}")
            
        except Exception as e:
            pytest.fail(f"Order validation failed: {e}")
    
    @pytest.mark.slow
    async def test_complete_trading_workflow_simulation(self, execution_engine):
        """Test complete trading workflow in simulation mode."""
        logger.info("Testing complete trading workflow simulation...")
        
        try:
            # Test portfolio setup
            portfolio_manager = execution_engine.portfolio_manager
            risk_manager = execution_engine.risk_manager
            
            # Set up test scenario
            test_symbols = ["BTCUSDT", "ETHUSDT"]
            
            # Update volatility data (simulated)
            for i, symbol in enumerate(test_symbols):
                portfolio_manager.update_volatility_data(symbol, 0.02 + i * 0.005)
            
            # Force rebalancing
            portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
            allocations = portfolio_manager.rebalance_portfolio(test_symbols)
            
            logger.info(f"Portfolio allocations: {len(allocations)} symbols")
            
            # Test position sizing for each allocation
            for symbol, allocation in allocations.items():
                # Get current market price
                ticker = await execution_engine.binance_client.get_ticker(symbol)
                market_price = float(ticker["price"])
                
                # Calculate position size
                position_result = risk_manager.calculate_position_size(
                    symbol=symbol,
                    allocated_capital=allocation.allocated_capital,
                    atr_value=portfolio_manager.get_volatility_ema(symbol),
                    entry_price=market_price
                )
                
                logger.info(f"{symbol}: Allocation ${allocation.allocated_capital:.2f} → Position ${position_result['size_usdt']:.2f}")
                
                # Verify calculations
                assert position_result['size_usdt'] > 0, f"Position size should be positive for {symbol}"
                assert position_result['size_usdt'] <= allocation.allocated_capital, \
                    f"Position should not exceed allocation for {symbol}"
                
                # Test SL/TP calculations
                sl_price, tp_price = risk_manager.calculate_stop_loss_take_profit(
                    entry_price=market_price,
                    side="buy",
                    atr_adjusted=portfolio_manager.get_volatility_ema(symbol) * market_price
                )
                
                # Verify SL/TP structure
                assert sl_price < market_price < tp_price, f"SL < Entry < TP for {symbol}"
                
                # Calculate risk-reward ratio
                risk = market_price - sl_price
                reward = tp_price - market_price
                rr_ratio = reward / risk
                
                logger.info(f"{symbol} RR ratio: {rr_ratio:.2f}:1")
                assert 1.8 <= rr_ratio <= 2.2, f"Risk-reward ratio should be ~2:1 for {symbol}"
            
        except Exception as e:
            pytest.fail(f"Complete workflow simulation failed: {e}")
    
    async def test_stress_scenarios_simulation(self, execution_engine):
        """Test stress scenarios in simulation mode."""
        logger.info("Testing stress scenarios simulation...")
        
        try:
            stress_handler = execution_engine.stress_handler
            
            # Test flash crash scenario
            test_symbol = "BTCUSDT"
            atr_value = 0.02
            flash_drop = 0.10  # 10% drop
            
            is_flash_crash = stress_handler.check_flash_crash(test_symbol, flash_drop, atr_value)
            assert is_flash_crash, "Should detect flash crash"
            logger.info("Flash crash detection: PASSED")
            
            # Test kill switch scenario
            high_drawdown = 0.16  # 16% drawdown
            should_trigger = stress_handler.should_trigger_kill_switch(high_drawdown)
            assert should_trigger, "Should trigger kill switch at 16% drawdown"
            logger.info("Kill switch detection: PASSED")
            
            # Test liquidity filters
            low_volume = 2_000_000  # $2M < $5M threshold
            high_spread = 0.002     # 0.2% > 0.15% threshold
            
            should_skip_volume = stress_handler.check_liquidity_filters(low_volume, 0.001)
            should_skip_spread = stress_handler.check_liquidity_filters(10_000_000, high_spread)
            
            assert should_skip_volume, "Should skip trading with low volume"
            assert should_skip_spread, "Should skip trading with high spread"
            logger.info("Liquidity filters: PASSED")
            
        except Exception as e:
            pytest.fail(f"Stress scenarios simulation failed: {e}")
    
    async def test_real_time_data_processing(self, binance_client):
        """Test real-time data processing capabilities."""
        logger.info("Testing real-time data processing...")
        
        test_symbols = ["BTCUSDT", "ETHUSDT"]
        
        try:
            # Test data retrieval for multiple symbols
            start_time = datetime.now()
            
            market_data = {}
            for symbol in test_symbols:
                ticker = await binance_client.get_ticker(symbol)
                orderbook = await binance_client.get_orderbook(symbol, limit=5)
                
                market_data[symbol] = {
                    "price": float(ticker["price"]),
                    "timestamp": datetime.now(),
                    "spread": self._calculate_spread(orderbook)
                }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Data processing time: {processing_time:.3f}s for {len(test_symbols)} symbols")
            
            # Verify processing speed requirement (<30ms per bar from document)
            max_processing_time = 0.03 * len(test_symbols)  # 30ms per symbol
            assert processing_time <= max_processing_time, \
                f"Processing too slow: {processing_time:.3f}s > {max_processing_time:.3f}s"
            
            # Verify data quality
            for symbol, data in market_data.items():
                assert data["price"] > 0, f"Price should be positive for {symbol}"
                assert data["spread"] >= 0, f"Spread should be non-negative for {symbol}"
                assert data["spread"] <= 0.01, f"Spread should be reasonable for {symbol}"
                
                logger.info(f"{symbol}: ${data['price']:.2f}, spread: {data['spread']:.4%}")
            
        except Exception as e:
            pytest.fail(f"Real-time data processing failed: {e}")
    
    def _calculate_spread(self, orderbook):
        """Calculate bid-ask spread from orderbook."""
        if not orderbook.get("bids") or not orderbook.get("asks"):
            return 0.0
        
        best_bid = float(orderbook["bids"][0][0])
        best_ask = float(orderbook["asks"][0][0])
        
        return (best_ask - best_bid) / best_bid
    
    async def test_error_handling_and_recovery(self, binance_client):
        """Test error handling and recovery mechanisms."""
        logger.info("Testing error handling and recovery...")
        
        try:
            # Test with invalid symbol
            try:
                await binance_client.get_ticker("INVALIDSYMBOL")
                pytest.fail("Should raise exception for invalid symbol")
            except Exception as e:
                logger.info(f"Correctly handled invalid symbol: {type(e).__name__}")
            
            # Test connection recovery after error
            # Verify system can continue after error
            ticker = await binance_client.get_ticker("BTCUSDT")
            assert ticker is not None, "Should recover and work normally after error"
            logger.info("Recovery after error: PASSED")
            
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")
    
    @pytest.mark.slow
    async def test_extended_monitoring_session(self, binance_client):
        """Test extended monitoring session for stability."""
        logger.info("Testing extended monitoring session...")
        
        test_symbol = "BTCUSDT"
        monitoring_duration = 60  # 60 seconds
        data_points = []
        
        try:
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < monitoring_duration:
                # Collect data point
                ticker = await binance_client.get_ticker(test_symbol)
                price = float(ticker["price"])
                timestamp = datetime.now()
                
                data_points.append({
                    "price": price,
                    "timestamp": timestamp
                })
                
                # Wait 1 second between requests
                await asyncio.sleep(1)
            
            # Analyze collected data
            logger.info(f"Collected {len(data_points)} data points over {monitoring_duration}s")
            
            # Verify data consistency
            assert len(data_points) >= monitoring_duration * 0.9, \
                "Should collect most data points successfully"
            
            # Check for reasonable price stability (no extreme jumps)
            prices = [dp["price"] for dp in data_points]
            price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] 
                           for i in range(1, len(prices))]
            
            max_change = max(price_changes) if price_changes else 0
            logger.info(f"Maximum price change: {max_change:.4%}")
            
            # Price changes should be reasonable (no >5% jumps in 1 second)
            assert max_change <= 0.05, f"Price changes too extreme: {max_change:.4%}"
            
        except Exception as e:
            pytest.fail(f"Extended monitoring session failed: {e}")


@pytest.mark.live
class TestLiveTestnetValidationSynchronous:
    """Synchronous tests for components that don't require async."""
    
    def test_testnet_configuration(self):
        """Test testnet configuration is properly set up."""
        # Verify testnet config exists
        assert hasattr(config, 'binance_futures_testnet'), "Testnet config should exist"
        
        testnet_config = config.binance_futures_testnet
        assert "testnet_api_key" in testnet_config, "Should have testnet API key"
        assert "testnet_api_secret" in testnet_config, "Should have testnet API secret"
        
        # Verify keys are not empty
        api_key = testnet_config["testnet_api_key"]
        api_secret = testnet_config["testnet_api_secret"]
        
        assert api_key and api_key.strip(), "API key should not be empty"
        assert api_secret and api_secret.strip(), "API secret should not be empty"
        assert len(api_key) >= 32, "API key should be reasonable length"
        assert len(api_secret) >= 32, "API secret should be reasonable length"
        
    def test_signal_generation_for_testnet(self):
        """Test signal generation suitable for testnet trading."""
        # Create realistic signal for testnet
        signal = TradeSignal(
            symbol="BTCUSDT",
            side="buy",
            action="open",
            strategy_id="testnet_validation",
            metadata={
                "confidence": 0.7,
                "entry_price": 45000.0,
                "timeframe": "1m",
                "strategy_params": {"ma_fast": 10, "ma_slow": 20}
            },
            signal_confidence=0.7
        )
        
        # Verify signal is suitable for testnet
        assert signal.symbol in ["BTCUSDT", "ETHUSDT", "XRPUSDT"], "Should use major trading pairs"
        assert signal.side in ["buy", "sell"], "Should have valid side"
        assert 0.5 <= signal.signal_confidence <= 1.0, "Should have reasonable confidence"
        assert signal.metadata["entry_price"] > 0, "Should have positive entry price"
        
    def test_portfolio_allocation_for_testnet(self):
        """Test portfolio allocation suitable for testnet capital."""
        # Use smaller capital for testnet
        testnet_capital = 1000.0  # $1000 for testing
        portfolio = ProductionPortfolioManager(total_capital=testnet_capital)
        
        test_symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Update volatilities
        for symbol in test_symbols:
            portfolio.update_volatility_data(symbol, 0.025)
        
        # Force rebalancing
        portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio.rebalance_portfolio(test_symbols)
        
        # Verify allocations are reasonable for testnet
        for symbol, allocation in allocations.items():
            assert allocation.allocated_capital >= 50.0, \
                f"Allocation should be at least $50 for {symbol}"
            assert allocation.allocated_capital <= testnet_capital * 0.5, \
                f"Single allocation should not exceed 50% for {symbol}"


def run_live_testnet_validation():
    """Run live testnet validation tests."""
    print("🌐 Starting Live Testnet Validation")
    print("=" * 60)
    
    # Check if testnet credentials are available
    try:
        import config
        if not hasattr(config, 'binance_futures_testnet'):
            print("❌ Testnet configuration not found. Skipping live tests.")
            return
        
        testnet_config = config.binance_futures_testnet
        if not testnet_config.get('testnet_api_key') or not testnet_config.get('testnet_api_secret'):
            print("❌ Testnet credentials not configured. Skipping live tests.")
            return
    except ImportError:
        print("❌ Config module not found. Skipping live tests.")
        return
    
    # Run the tests
    pytest_args = [
        __file__,
        "-v",
        "-m", "live",
        "--tb=short"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ All live testnet validation tests passed!")
    else:
        print("❌ Some live testnet validation tests failed.")
    
    return exit_code


if __name__ == "__main__":
    run_live_testnet_validation()
