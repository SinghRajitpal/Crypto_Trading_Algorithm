"""Loader utilities for manifest-based Layer A GRU forecasters."""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import config
from algorithm.forecast.gru_torch import GRUForecasterTorch
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LayerAModel:
    symbol: str
    timeframe: str
    lookback: int
    retrain_days: int
    artifact_dir: str
    forecaster: GRUForecasterTorch

    def predict(self, window: np.ndarray) -> Tuple[float, Optional[float], float]:
        """Return log-return prediction, optional sigma (None), and latency_ms."""
        start = time.time()
        mu_log = self.forecaster.predict_log_return(window)
        latency_ms = (time.time() - start) * 1000.0
        return mu_log, None, latency_ms


def load_manifest(manifest_path: str, device: Optional[str] = None) -> Dict[str, LayerAModel]:
    """Load all Layer A models defined in a manifest JSON list."""
    if not manifest_path or not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        entries = json.load(f)
    models: Dict[str, LayerAModel] = {}
    for entry in entries:
        try:
            sym = entry["symbol"].upper()
            artifact_dir = entry["artifact_dir"]
            model_dir = _resolve_model_dir(artifact_dir)
            lookback = int(entry.get("lookback", config.LAYERA_LOOKBACK))
            timeframe = entry.get("timeframe", config.LAYERA_TIMEFRAME)
            retrain_days = int(entry.get("retrain_days", config.LAYERA_RETRAIN_DAYS))
            forecaster = GRUForecasterTorch.load(model_dir, device=device or config.GRU_DEVICE)
            models[sym] = LayerAModel(
                symbol=sym,
                timeframe=timeframe,
                lookback=lookback,
                retrain_days=retrain_days,
                artifact_dir=model_dir,
                forecaster=forecaster,
            )
        except Exception as exc:
            logger.warning("Failed to load Layer A model for %s: %s", entry.get("symbol"), exc)
            continue
    if not models:
        raise RuntimeError(f"No models loaded from manifest: {manifest_path}")
    logger.info("Loaded %d Layer A models from %s", len(models), manifest_path)
    return models


def _resolve_model_dir(artifact_dir: str) -> str:
    """Locate the directory containing model.pt for a Layer A artifact.

    Prefers checkpoints/step_XXX/model.pt with the highest step, else artifact root.
    """
    root_model = os.path.join(artifact_dir, "model.pt")
    if os.path.exists(root_model):
        return artifact_dir

    ckpt_glob = os.path.join(artifact_dir, "checkpoints", "step_*", "model.pt")
    candidates = []
    for path in sorted(glob.glob(ckpt_glob)):
        step = -1
        try:
            step_str = os.path.basename(os.path.dirname(path)).replace("step_", "")
            step = int(step_str)
        except Exception:
            step = -1
        candidates.append((step, path))
    if candidates:
        # Pick highest step number; if parsing failed, sorted order still deterministic.
        candidates.sort(key=lambda x: x[0])
        chosen = candidates[-1][1]
        chosen_dir = os.path.dirname(chosen)
        logger.info("Resolved model dir to latest checkpoint: %s", chosen_dir)
        return chosen_dir

    raise FileNotFoundError(f"Missing model weights: {root_model}")
