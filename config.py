# Binance API configuration
from utils.logging_config import get_logger, console_log

# Get logger for this module
logger = get_logger(__name__)

# Log configuration loading
console_log("Loading trading algorithm configuration")

binance_futures = {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
}

binance_futures_testnet = {
    "testnet_api_key": "74384890789851e3ce8a1b021079111d950dcc96da8ca279d129a1e987820fec",
    "testnet_api_secret": "ba10816594558c8cd93d04e823608f408016aa15c4d632a2214ca950a50b1967",
}

# Symbol universe and timeframe settings
PRIMARY_TIMEFRAME = "5m"
DEFAULT_UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "SOLUSDT"
]

# Symbol-timeframe pairs to monitor
symbols = [(symbol, PRIMARY_TIMEFRAME) for symbol in DEFAULT_UNIVERSE]

# =============================================================================
# TRADING ALGORITHM PARAMETERS
# =============================================================================

# Universe Selection Parameters
UNIVERSE_MAX_RANK = 50  # Maximum rank to include from market-cap list
UNIVERSE_MIN_DOLLAR_VOLUME = 20_000_000  # Minimum 30-day median daily dollar volume
UNIVERSE_LOOKBACK_DAYS = 30  # Lookback for median volume calculation

# Bar Validation Parameters
BAR_RETURN_ABS_THRESHOLD = 0.25  # Reject bars if simple return exceeds 25%
MIN_BAR_VOLUME = 1.0  # Minimum raw volume units for a bar to be considered valid

# Regression / Forecast Parameters
REGRESSION_WINDOW = 120  # Bars used for rolling regression
REGRESSION_FEATURE_MODE = "log_price"  # Predictor type for linear model

# Risk Model Data Parameters
RISK_WINDOW = 180  # Bars used for covariance estimation
EWMA_LAMBDA = 0.94  # Decay for EWMA volatility calculations
COVARIANCE_SHRINKAGE = 0.6  # Alpha for sample-vs-target shrinkage

# Mean-Variance Optimizer Parameters
RISK_AVERSION = 3.0  # Higher = more risk averse
WEIGHT_MIN = -0.3  # Minimum per-asset weight (allows mild shorts)
WEIGHT_MAX = 0.3  # Maximum per-asset weight
MAX_NET_EXPOSURE = 0.25  # Absolute net exposure cap
MAX_GROSS_EXPOSURE = 1.2  # Sum |w| cap
CONTRACT_MULTIPLIER = 1.0  # Futures contract multiplier
MIN_ORDER_NOTIONAL = 10.0  # Skip orders smaller than $10

# Configuration validation and logging
def validate_config():
    """Validate critical configuration parameters and log warnings if needed."""
    
    # Check API keys
    testnet_key = binance_futures_testnet.get('testnet_api_key')
    if not testnet_key or testnet_key == 'your_testnet_api_key':
        logger.warning("Testnet API key not configured - using default placeholder")
    else:
        logger.info("Testnet API credentials configured")
    
    # Validate symbol configuration
    logger.info(f"Configured {len(symbols)} trading symbols: {[s[0] for s in symbols]}")

# Run validation when module is imported
validate_config()
