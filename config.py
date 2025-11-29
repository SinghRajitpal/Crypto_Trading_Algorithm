# GRU-based trading configuration for Binance USD-M futures (8h timeframe).

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
PRIMARY_TIMEFRAME = "8h"
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
# Embargo between train/val splits to avoid leakage (in bars)
REGRESSION_EMBARGO_BARS = 10
# GRU Forecast Parameters (Keras baseline)
GRU_TIMEFRAME = "8h"
GRU_LOOKBACK = 64
GRU_FEATURES = ["log_return", "log_volume"]
GRU_HIDDEN_SIZE = 64
GRU_NUM_LAYERS = 1
GRU_DROPOUT = 0.0
GRU_BATCH_SIZE = 32
GRU_LR = 1e-3
GRU_EPOCHS = 50
GRU_EARLY_STOP_PATIENCE = 5
GRU_GRAD_CLIP = 1.0
GRU_VALIDATION_SPLIT = 0.2
GRU_HUBER_DELTA = 1.0
GRU_MODEL_DIR = "backtest/backtest_results/gru_artifacts"
GRU_TRAIN_VERBOSE = 1  # Keras fit verbosity (0=silent, 1=progress bar)
GRU_DEVICE = "mps"  # Try Apple Metal first, fallback to cpu
GRU_RETRAIN_DAYS = 30
GRU_MIN_TRAIN_SAMPLES = 200

# Manifest-driven Layer A GRU integration
LAYERA_MANIFEST_PATH = "backtest/backtest_results/2025-11-29_trial02/layerA_artifacts/layerA_manifest.json"  # e.g., "backtest/backtest_results/2025-11-28_trial01/layerA_artifacts/layerA_manifest.json"
LAYERA_TIMEFRAME = GRU_TIMEFRAME
LAYERA_LOOKBACK = GRU_LOOKBACK
LAYERA_RETRAIN_DAYS = GRU_RETRAIN_DAYS
LAYERA_MIN_TRAIN_SAMPLES = GRU_MIN_TRAIN_SAMPLES
# Forecast behavior
FORECAST_FALLBACK = "legacy_gru"  # "none" or "legacy_gru"
FORECAST_LATENCY_WARN_MS = 500.0
FORECAST_SIGMA_WARN = None  # optional cap on sigma for filtering, set to float to enable
FORECAST_DIVERGENCE_WARN = 0.02  # abs(pred - realized) threshold for warning
FORECAST_LOG_PATH = None  # optional per-bar forecast log CSV in live/demo

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
