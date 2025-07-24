# Binance API configuration
binance_futures = {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
}

binance_futures_testnet = {
    "testnet_api_key": "74384890789851e3ce8a1b021079111d950dcc96da8ca279d129a1e987820fec",
    "testnet_api_secret": "ba10816594558c8cd93d04e823608f408016aa15c4d632a2214ca950a50b1967",
}

# Symbol-timeframe pairs to monitor
symbols = [
    ("BTCUSDT", "1m"),
    ("ETHUSDT", "1m"),
    ("XRPUSDT", "1m"),
    ("BNBUSDT", "1m"),
    ("SOLUSDT", "1m")
]

# =============================================================================
# TRADING ALGORITHM PARAMETERS
# =============================================================================

# Portfolio Management Parameters
TARGET_VOLATILITY = 0.18  # 18% target portfolio volatility
MAX_ALLOCATION_PCT = 0.85  # 85% maximum capital allocation
ALPHA_CORRELATION = 0.3  # Correlation adjustment parameter for weight calculation
LOOKBACK_BARS = 60  # EMA lookback period for volatility and correlation
REGIME_PERCENTILE = 75  # 75th percentile threshold for high volatility regime
REBALANCE_HOURS = 24  # Hours between portfolio rebalancing

# Risk Management Parameters
RISK_PER_TRADE_PCT = 0.008  # 0.8% risk per trade
KELLY_FRACTION = 0.7  # Fractional Kelly criterion (70%)
MAX_LEVERAGE = 20  # Maximum leverage cap
MIN_ATR_FLOOR = 0.001  # Minimum ATR floor to prevent excessive sizing

# Cost Parameters - Comprehensive Trading Costs
BASE_TRADING_FEE_PCT = 0.0004  # 0.04% base trading fee (Binance futures)
BASE_SPREAD_PCT = 0.0010  # 0.10% typical spread estimate
BASE_SLIPPAGE_PCT = 0.0003  # 0.03% market impact slippage
BASE_COMMISSION_PCT = 0.0001  # 0.01% additional commission/platform costs
FUNDING_RATE_8H_PCT = 0.0001  # 0.01% typical 8-hour funding cost

# Total base cost: 0.04% + 0.10% + 0.03% + 0.01% + 0.01% = 0.19%
BASE_COST_PCT = BASE_TRADING_FEE_PCT + BASE_SPREAD_PCT + BASE_SLIPPAGE_PCT + BASE_COMMISSION_PCT + FUNDING_RATE_8H_PCT
VOLATILITY_COST_MULTIPLIER = 0.5  # Cost adjustment multiplier for volatility

# Stop Loss and Take Profit Parameters
ATR_STOP_MULTIPLIER = 1.8  # Stop loss = Entry ± 1.8×ATR
ATR_TRAIL_MULTIPLIER = 0.8  # Trailing stop = 0.8×ATR
RISK_REWARD_RATIO = 2.0  # Take profit = Entry ± 2×|Entry-SL| (1:2 risk-reward)
PARTIAL_EXIT_RATIO = 0.4  # 40% partial exit at 1:1 risk-reward

# Test and Validation Parameters
TEST_VOLATILITY_SAMPLE_SIZE = 25  # Number of samples for building volatility history
MAX_SCALING_MULTIPLIER = 0.6  # Maximum scaling multiplier in high volatility
MIN_ALLOCATION_THRESHOLD = 10000  # Minimum expected allocation for validation
MAX_ALLOCATION_THRESHOLD = 13000  # Maximum expected allocation for validation

# Default Test Symbols and Data
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TEST_VOLATILITIES = [0.015, 0.025, 0.020, 0.018, 0.030]

