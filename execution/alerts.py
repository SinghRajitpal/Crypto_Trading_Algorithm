from typing import Dict, Any

import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


def check_thresholds(result: Dict[str, Any]) -> None:
    """Log warnings when monitored metrics exceed configured thresholds."""
    risk_cov_loss = result.get("risk_cov_loss", {})
    if risk_cov_loss:
        mse = risk_cov_loss.get("cov_port_mse")
        if mse is not None and mse > config.MONITOR_COV_MSE_WARN:
            logger.warning("ALERT: Covariance MSE exceeded threshold: %.4f", mse)

    risk_diag = result.get("risk_diag", {})
    vol_mse = risk_diag.get("vol_mse_avg")
    if vol_mse is not None and vol_mse > config.MONITOR_VOL_MSE_WARN:
        logger.warning("ALERT: Vol MSE exceeded threshold: %.4f", vol_mse)
