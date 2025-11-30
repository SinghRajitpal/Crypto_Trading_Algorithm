"""Manifest-driven Layer A GRU forecast strategy for live/demo use."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np

import config
from algorithm.forecast.forecast_result import ForecastResult
from algorithm.forecast.layera_loader import LayerAModel, load_manifest
from utils.logging_config import get_logger

logger = get_logger(__name__)


class LayerAForecastStrategy:
    """Strategy that serves per-symbol GRU forecasts from manifest artifacts."""

    strategy_id = "layerA_manifest_gru"

    def __init__(self, data_engine, manifest_path: str, symbols=None, timeframe: Optional[str] = None):
        if not manifest_path or not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Layer A manifest not found: {manifest_path}")
        self.data_engine = data_engine
        self.manifest_path = manifest_path
        self.timeframe = timeframe or config.LAYERA_TIMEFRAME
        self.symbols = [s.upper() for s in symbols] if symbols else None
        self.models: Dict[str, LayerAModel] = load_manifest(manifest_path, device=config.GRU_DEVICE)
        self.lookback_map = {sym: model.lookback for sym, model in self.models.items()}
        self.model_version = os.path.basename(os.path.dirname(manifest_path)) or manifest_path

    async def calculate_forecast(self) -> ForecastResult | None:
        # Ingest the latest bars into rolling windows
        self.data_engine.process_all_latest_bars(self.timeframe)
        universe = self.symbols or self.data_engine.get_active_universe()

        expected_returns: Dict[str, float] = {}
        latency_ms_list: List[float] = []
        sigma_list: List[float] = []
        ready = 0

        for sym in universe:
            sym_u = sym.upper()
            model = self.models.get(sym_u)
            if not model:
                continue
            if self.data_engine.get_missing_bars(sym_u, self.timeframe):
                continue
            if hasattr(self.data_engine, "feature_ready") and not self.data_engine.feature_ready(sym_u, model.lookback):
                logger.debug("Features not ready for %s | lookback=%d", sym_u, model.lookback)
                continue
            window = self._build_window(sym_u, model.lookback)
            if window is None:
                logger.debug("No feature window for %s | lookback=%d", sym_u, model.lookback)
                continue
            ready += 1
            try:
                if not np.all(np.isfinite(window)):
                    logger.warning("Non-finite values in feature window for %s; skipping", sym_u)
                    continue
                if getattr(model.forecaster, "feature_schema", None):
                    expected = len(model.forecaster.feature_schema)
                    if window.shape[1] != expected:
                        logger.warning(
                            "Feature window width mismatch for %s | expected=%d got=%d",
                            sym_u,
                            expected,
                            window.shape[1],
                        )
                        continue
                mu_log, sigma, latency_ms = model.predict(window)
                mu_simple = float(np.exp(mu_log) - 1.0)
                # Optional sigma filtering
                if config.FORECAST_SIGMA_WARN is not None and sigma is not None and sigma > config.FORECAST_SIGMA_WARN:
                    logger.warning("Skipping %s due to high sigma=%.6f", sym_u, sigma)
                    continue
                expected_returns[sym_u] = mu_simple
                latency_ms_list.append(latency_ms)
                if sigma is not None:
                    sigma_list.append(float(sigma))
                if latency_ms > config.FORECAST_LATENCY_WARN_MS:
                    logger.warning("High forecast latency | symbol=%s | latency_ms=%.1f", sym_u, latency_ms)
            except Exception as exc:
                logger.warning("Layer A forecast failed for %s: %s", sym_u, exc)
                continue

        if not expected_returns:
            return None

        latest_ts = 0
        for sym in expected_returns.keys():
            candle = self.data_engine.get_latest_candle(sym, self.timeframe)
            if candle:
                latest_ts = max(latest_ts, int(candle[0]))
        timestamp = latest_ts or int(time.time() * 1000)

        diagnostics = {
            "model_version": self.model_version,
            "latency_ms_avg": float(np.mean(latency_ms_list)) if latency_ms_list else None,
            "latency_ms_max": max(latency_ms_list) if latency_ms_list else None,
            "sigma_avg": float(np.mean(sigma_list)) if sigma_list else None,
            "sigma_max": max(sigma_list) if sigma_list else None,
            "coverage": len(expected_returns) / max(1, len(universe)),
            "ready": ready,
            "requested": len(universe),
        }

        return ForecastResult(
            timestamp=timestamp,
            universe=list(expected_returns.keys()),
            expected_returns=expected_returns,
            betas={},
            diagnostics=diagnostics,
        )

    def _build_window(self, symbol: str, lookback: int):
        """Construct the latest window of pre-computed features."""
        if not hasattr(self.data_engine, "get_feature_window"):
            return None
        window = self.data_engine.get_feature_window(symbol, lookback)
        if window is None:
            return None
        model = self.models.get(symbol)
        if model and getattr(model.forecaster, "feature_schema", None):
            expected = len(model.forecaster.feature_schema)
            if window.shape[1] != expected:
                logger.warning(
                    "Feature window width mismatch for %s | expected=%d got=%d",
                    symbol,
                    expected,
                    window.shape[1],
                )
                return None
        return window
