# Mean-variance trading configuration for 5-minute Binance USD-M futures.

# Binance API configuration
from utils.logging_config import get_logger, console_log

# Get logger for this module
logger = get_logger(__name__)

# Log configuration loading
console_log("Loading trading algorithm configuration")

binance_futures = {
    "api_key": "key",
    "api_secret": "secret",
}

binance_futures_testnet = {
    "testnet_api_key": "SMhgP5bwMLcMGkpEPI6Rh8pDBy4drNOjIVclhPEi8cNdbNATh1B48UHlv1timYMg",
    "testnet_api_secret": "ykdDkx7b16Isz3IV0Y5YU6BttVaIdEXM7ToVNIcNFeBcteVnkvppzdfTKluRszYd",
}

# Symbol universe and timeframe settings
PRIMARY_TIMEFRAME = "1h"
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
UNIVERSE_MAX_RANK = 20  # Maximum rank to include from market-cap list
UNIVERSE_MIN_DOLLAR_VOLUME = 20_000_000  # Minimum 30-day median daily dollar volume
UNIVERSE_LOOKBACK_DAYS = 30  # Lookback for median volume calculation
UNIVERSE_REFRESH_HOURS = 24  # Interval for refreshing universe ranks/market caps
BAR_GRID_TIMEFRAME = "1h"  # Global bar grid timeframe

# Bar Validation Parameters
BAR_RETURN_ABS_THRESHOLD = 0.25  # Reject bars if simple return exceeds 25%
MIN_BAR_VOLUME = 1.0  # Minimum raw volume units for a bar to be considered valid
OUTLIER_SIGMA_THRESHOLD = 8.0  # Sigma cap for raw log-return outlier flagging

# Regression / Forecast Parameters
REGRESSION_WINDOW = 120  # Bars used for rolling regression
REGRESSION_FEATURE_MODE = "log_price"  # Predictor type for linear model
LOG_RETURN_LAGS = [1, 3, 6, 12]  # Lags of log returns used as features
VOLUME_LAGS = [1, 3, 6, 12]  # Lags of volume used as features
MOMENTUM_WINDOWS = [3, 6, 12]  # Windowed log-return momentum features
VOL_WINDOWS = [6, 12, 24]  # Rolling realized variance windows (bars)
RANGE_WINDOWS = [6, 12]  # Rolling average high-low range windows
TURNOVER_WINDOWS = [6, 12]  # Rolling average volume windows
INCLUDE_TIME_OF_DAY = True  # Add time-of-day sin/cos features
REGRESSION_MAX_BARS = None  # Use all available history per asset (no cap)
REGRESSION_MIN_TRAIN = 120  # Minimum training observations required (must be <= buffer capacity)
REGRESSION_VAL_WINDOW = 500  # Validation window size for rolling CV blocks
# Training lookback horizon; if None, use all available history. Otherwise, use this many days before backtest start.
RIDGE_TRAIN_LOOKBACK_DAYS = 120
# Embargo between train/val splits to avoid leakage (in bars)
REGRESSION_EMBARGO_BARS = 10
# Candidate timeframes for Layer A selection (global TF)
TF_CANDIDATES = ["1m", "5m", "15m", "30m", "1h", "4h", "8h", "1d", "1w"]
# Per-timeframe training window candidates (bars); fallback to a default list if TF missing
W_CANDIDATES_BY_TF = {
    "1m": [200, 500, 1000],
    "5m": [250, 750, 1500],
    "15m": [250, 750, 1500],
    "30m": [500, 1500, 3000],
    "1h": [500, 2000, 4000],
    "4h": [300, 800, 1500],
    "8h": [300, 800, 1500],
    "1d": [200, 500, 1000],
    "1w": [100, 200, 300],
}

# Timeframe-specific lookbacks (days) for historical fetch; None means all available.
TIMEFRAME_LOOKBACK_DAYS = {
    "1m": 90,
    "5m": 180,
    "10m": 180,
    "30m": 365,
    "1h": 730,
    "4h": 1095,
    "8h": 1460,
    "1d": 1825,
    "1w": 1825,
}
# Candidate historical lookback windows (days) for Layer A per timeframe; None means use all cached history.
LAYERA_LOOKBACK_DAYS_GRID = {
    "1m": [30, 90],
    "5m": [60, 180],
    "15m": [90, 270],
    "30m": [180],
    "1h": [180, 365],
    "4h": [365, 730],
    "8h": [365, 730],
    "1d": [730],
    "1w": [1825],
}
# Optional Layer-A only caps on bars per timeframe to speed CV (None means full history)
LAYERA_MAX_BARS_BY_TF = {
    # Keep the smallest timeframes on a tighter leash to avoid ballooning memory.
    "1m": 50_000,
    "5m": 30_000,
    "15m": 20_000,
    # Higher timeframes are already sparse; clip only if we hit memory walls.
    "30m": 15_000,
    "1h": None,
    "4h": None,
    "8h": None,
    "1d": None,
    "1w": None,
}
LAYERA_DEBUG_MEM = True  # Enable verbose Layer A memory diagnostics
# Optional Layer-A only cap on regression bars (applied before CV); None means no additional cap
LAYERA_REGRESSION_MAX_BARS = 20_000
RIDGE_T_THRESHOLD = 1.2  # Default |t| threshold for ridge feature pruning
RIDGE_K_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4]

# Risk Model Data Parameters
RISK_WINDOW = 180  # Bars used for covariance estimation
EWMA_LAMBDA = 0.94  # Decay for EWMA volatility calculations
COVARIANCE_SHRINKAGE = 0.6  # Alpha for sample-vs-target shrinkage
RISK_COV_WINDOW = 250  # Window for sample covariance
RISK_USE_GARCH = True  # Enable GARCH(1,1) variance candidate
GARCH_PARAMS = {"omega": 1e-6, "alpha": 0.05, "beta": 0.9}
RISK_DIAG_WINDOW = 200  # Window length for diagnostics (MALV, portfolio losses)

# Mean-Variance Optimizer Parameters
RISK_AVERSION = 3.0  # Higher = more risk averse
WEIGHT_MIN = -0.3  # Minimum per-asset weight (allows mild shorts)
WEIGHT_MAX = 0.3  # Maximum per-asset weight
MAX_NET_EXPOSURE = 0.25  # Absolute net exposure cap
MAX_GROSS_EXPOSURE = 1.2  # Sum |w| cap
CONTRACT_MULTIPLIER = 1.0  # Futures contract multiplier
MIN_ORDER_NOTIONAL = 5.0  # Binance USD-M min notional is typically ~$5
CONTRACT_MULTIPLIER_PER_ASSET = {}  # Optional per-asset contract multipliers
TURNOVER_PENALTY_LAMBDA = 0.1  # Quadratic turnover penalty weight

# Impact / Execution Cost Parameters
IMPACT_KAPPA_DEFAULT = 0.01  # Temporary impact cost coefficient (concave power-law)
IMPACT_KAPPA_OVERRIDES = {}  # Optional per-asset kappa overrides
IMPACT_DELTA = 0.5  # Concavity for power-law impact
IMPACT_PROPAGATOR_DECAY = 0.5  # Exponential decay for impact propagator

# Kelly Overlay Parameters
KELLY_FRACTION_BASE = 0.5  # Fractional Kelly base lambda
DRAWDOWN_THRESHOLDS = [0.10, 0.20]  # Drawdown levels to scale Kelly
DRAWDOWN_LAMBDAS = [0.5, 0.25]  # Multipliers corresponding to thresholds
KELLY_MAX_LEVERAGE = 2.0  # Cap on Kelly leverage multiplier
KELLY_VOL_THRESHOLD = 0.05  # Volatility level to start dampening Kelly further

# Monitoring thresholds
MONITOR_VOL_MSE_WARN = 0.05  # Warn if avg vol MSE exceeds this
MONITOR_COV_MSE_WARN = 0.05  # Warn if cov portfolio MSE exceeds this
MONITOR_LOG_PATH = "logs/monitoring.jsonl"  # File to persist monitoring records
BENCHMARK_SYMBOL = "BTCUSDT"  # Default benchmark for backtests/monitoring

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
