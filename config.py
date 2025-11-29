# GRU-based trading configuration for Binance USD-M futures (8h timeframe).

from utils.logging_config import get_logger, console_log

logger = get_logger(__name__)
console_log("Loading trading algorithm configuration")

binance_futures = {
    "api_key": "key",
    "api_secret": "secret",
}

binance_futures_demo = {
    "demo_api_key": "SMhgP5bwMLcMGkpEPI6Rh8pDBy4drNOjIVclhPEi8cNdbNATh1B48UHlv1timYMg",
    "demo_api_secret": "ykdDkx7b16Isz3IV0Y5YU6BttVaIdEXM7ToVNIcNFeBcteVnkvppzdfTKluRszYd",
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

symbols = [(symbol, PRIMARY_TIMEFRAME) for symbol in DEFAULT_UNIVERSE]

# Universe Selection Parameters
UNIVERSE_MAX_RANK = 20
UNIVERSE_MIN_DOLLAR_VOLUME = 20_000_000
UNIVERSE_LOOKBACK_DAYS = 30
UNIVERSE_REFRESH_HOURS = 24
BAR_GRID_TIMEFRAME = "1h"

# Bar Validation Parameters (unused)
BAR_RETURN_ABS_THRESHOLD = 0.25
MIN_BAR_VOLUME = 1.0
OUTLIER_SIGMA_THRESHOLD = 8.0

# Regression / Forecast Parameters
REGRESSION_WINDOW = 120
REGRESSION_FEATURE_MODE = "log_price"
LOG_RETURN_LAGS = [1, 3, 6, 12]
VOLUME_LAGS = [1, 3, 6, 12]
MOMENTUM_WINDOWS = [3, 6, 12]
VOL_WINDOWS = [6, 12, 24]
RANGE_WINDOWS = [6, 12]
TURNOVER_WINDOWS = [6, 12]
INCLUDE_TIME_OF_DAY = True
REGRESSION_MAX_BARS = None
REGRESSION_MIN_TRAIN = 120
REGRESSION_VAL_WINDOW = 500
REGRESSION_EMBARGO_BARS = 10

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
GRU_TRAIN_VERBOSE = 1
GRU_DEVICE = "mps"
GRU_RETRAIN_DAYS = 30
GRU_MIN_TRAIN_SAMPLES = 200

# Risk Model Data Parameters
RISK_WINDOW = 180
EWMA_LAMBDA = 0.94
COVARIANCE_SHRINKAGE = 0.6
RISK_COV_WINDOW = 250
RISK_USE_GARCH = True
GARCH_PARAMS = {"omega": 1e-6, "alpha": 0.05, "beta": 0.9}
RISK_DIAG_WINDOW = 200

# Mean-Variance Optimizer Parameters
RISK_AVERSION = 3.0
WEIGHT_MIN = -0.3
WEIGHT_MAX = 0.3
MAX_NET_EXPOSURE = 0.25
MAX_GROSS_EXPOSURE = 1.2
CONTRACT_MULTIPLIER = 1.0
MIN_ORDER_NOTIONAL = 5.0
CONTRACT_MULTIPLIER_PER_ASSET = {}
TURNOVER_PENALTY_LAMBDA = 0.1

# Impact / Execution Cost Parameters
IMPACT_KAPPA_DEFAULT = 0.01
IMPACT_KAPPA_OVERRIDES = {}
IMPACT_DELTA = 0.5
IMPACT_PROPAGATOR_DECAY = 0.5

# Kelly Overlay Parameters
KELLY_FRACTION_BASE = 0.5
DRAWDOWN_THRESHOLDS = [0.10, 0.20]
DRAWDOWN_LAMBDAS = [0.5, 0.25]
KELLY_MAX_LEVERAGE = 2.0
KELLY_VOL_THRESHOLD = 0.05

# Monitoring thresholds
MONITOR_VOL_MSE_WARN = 0.05
MONITOR_COV_MSE_WARN = 0.05
MONITOR_LOG_PATH = "logs/monitoring.jsonl"
BENCHMARK_SYMBOL = "BTCUSDT"

# Layer A manifest/forecast settings
LAYERA_MANIFEST_PATH = "backtest/backtest_results/2025-11-29_trial02/layerA_artifacts/layerA_manifest.json"
LAYERA_TIMEFRAME = GRU_TIMEFRAME
LAYERA_LOOKBACK = GRU_LOOKBACK
LAYERA_RETRAIN_DAYS = GRU_RETRAIN_DAYS
LAYERA_MIN_TRAIN_SAMPLES = GRU_MIN_TRAIN_SAMPLES
FORECAST_FALLBACK = "legacy_gru"
FORECAST_LATENCY_WARN_MS = 500.0
FORECAST_SIGMA_WARN = None
FORECAST_DIVERGENCE_WARN = 0.02
FORECAST_LOG_PATH = None

def validate_config():
    demo_key = binance_futures_demo.get('demo_api_key')
    if not demo_key or demo_key == 'your_demo_api_key':
        logger.warning("Demo API key not configured - using default placeholder")
    else:
        logger.info("Demo API credentials configured")
    logger.info(f"Configured {len(symbols)} trading symbols: {[s[0] for s in symbols]}")

validate_config()
