# Centralized knobs for live/demo trading and backtesting.
# GRU-based trading configuration for Binance USD-M futures (8h timeframe).

from utils.logging_config import get_logger, console_log

logger = get_logger(__name__)
console_log("Loading trading algorithm configuration")

binance_futures = {
    "api_key": "key",
    "api_secret": "secret",
}

binance_futures_demo = {
    "demo_api_key": "demo_key",
    "demo_api_secret": "demo_secret",
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

GRU_TIMEFRAME = "8h"
GRU_LOOKBACK = 64
GRU_FEATURE_SCHEMA = [
    "log_return",
    "log_volume",
    "tsmom_fast_vol_z",
    "tsmom_med_vol_z",
    "rsi12_innov_z",
    "atrpct_innov_z",
    "funding_z",
    "ofi_z",  # fallback to obvtilt_z if OFI unavailable
]
GRU_HIDDEN_SIZE = 64
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.2
GRU_BATCH_SIZE = 32
GRU_LR = 1e-3
GRU_EPOCHS = 100
GRU_EARLY_STOP_PATIENCE = 8
GRU_GRAD_CLIP = 1.0
GRU_VALIDATION_SPLIT = 0.2
GRU_LOSS = "mse"  # options: mse, huber
GRU_OPTIMIZER = "adam"
GRU_BIDIRECTIONAL = False
GRU_MODEL_DIR = "backtest/backtest_results/gru_artifacts"
GRU_TRAIN_VERBOSE = 1
GRU_DEVICE = "mps"
GRU_RETRAIN_DAYS = 30
GRU_MIN_TRAIN_SAMPLES = 200

# Feature engineering knobs for GRU pipeline
FEATURE_WINSOR_PCT = 0.01
FEATURE_Z_WARMUP = 200
FEATURE_ADV_WINDOW = 60
FEATURE_FALLBACK_OBV = True
FEATURE_ATR_WINDOW = 14
FEATURE_ATR_EMA = 20
FEATURE_RSI_WINDOW = 12
FEATURE_RSI_EMA = 20
FEATURE_TSMOM_FAST_K = 3
FEATURE_TSMOM_MED_K = 15

# Risk Model Data Parameters
RISK_WINDOW = 180
EWMA_LAMBDA = 0.94
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

# Execution Cost Parameters (simplified slippage + fees)
FEE_RATE = 0.0004  # per-side taker fee as fraction of notional
SLIPPAGE_BPS_DEFAULT = 3.0
SLIPPAGE_BPS_OVERRIDES = {}
# Optional per-bar cost budget (basis points of NAV). None disables the budget guard.
COST_BUDGET_BP = None

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
LAYERA_MANIFEST_PATH = "backtest/backtest_results/e2e_run/layerA_artifacts/layerA_manifest.json" #"backtest/backtest_results/2025-11-29_trial02/layerA_artifacts/layerA_manifest.json"
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
