"""Layer A trainer for GRU-based forecaster."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import config
from algorithm.forecast.gru_torch import GRUForecasterTorch, GRUSpec
from data.gru_sequence_builder import build_sequences
from data.historical_data import HistoricalDataFetcher
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GRULayerResult:
    model_dir: str
    timeframe: str
    lookback: int
    samples: int
    val_loss: Optional[float] = None
    train_start: Optional[str] = None
    train_end: Optional[str] = None

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "GRULayerResult":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class GRULayerTrainer:
    """Train a single-asset GRU forecaster on historical OHLCV."""

    def __init__(
        self,
        symbol: str,
        timeframe: str = config.GRU_TIMEFRAME,
        lookback: int = config.GRU_LOOKBACK,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.lookback = lookback

    async def train(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        output_dir: str,
    ) -> GRULayerResult:
        os.makedirs(output_dir, exist_ok=True)
        fetcher = HistoricalDataFetcher(testnet=False)
        try:
            # Prefer cache-only mode to avoid network hangs when cache is available.
            df = await fetcher.download_ohlcv(self.symbol, self.timeframe, start, end, force=False, cache_only=True)
        finally:
            await fetcher.close()

        if df is None or df.empty:
            raise ValueError("No historical data available for GRU training.")

        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        X, y = build_sequences(df, lookback=self.lookback)
        if X.size == 0 or y.size == 0:
            raise ValueError("Insufficient data to build GRU training sequences.")

        forecaster = GRUForecasterTorch(
            lookback=self.lookback,
            input_size=2,
            hidden_size=config.GRU_HIDDEN_SIZE,
            num_layers=config.GRU_NUM_LAYERS,
            dropout=config.GRU_DROPOUT,
            learning_rate=config.GRU_LR,
            huber_delta=config.GRU_HUBER_DELTA,
            grad_clip=config.GRU_GRAD_CLIP,
            batch_size=config.GRU_BATCH_SIZE,
            epochs=config.GRU_EPOCHS,
            patience=config.GRU_EARLY_STOP_PATIENCE,
            validation_split=config.GRU_VALIDATION_SPLIT,
            verbose=config.GRU_TRAIN_VERBOSE,
            device=config.GRU_DEVICE,
        )
        history, spec = forecaster.fit(X, y)
        spec.train_start = str(df.index.min())
        spec.train_end = str(df.index.max())
        if "val_loss" in history and history["val_loss"]:
            spec.val_loss = float(history["val_loss"][-1])
            spec.epochs = len(history["val_loss"])
        model_dir = os.path.join(output_dir, "gru_model")
        forecaster.save(model_dir, spec)

        result = GRULayerResult(
            model_dir=model_dir,
            timeframe=self.timeframe,
            lookback=self.lookback,
            samples=int(len(y)),
            val_loss=spec.val_loss,
            train_start=spec.train_start,
            train_end=spec.train_end,
        )
        result_path = os.path.join(output_dir, "gru_spec.json")
        result.to_json(result_path)
        logger.info("GRU Layer A complete | model_dir=%s | samples=%d | val_loss=%s", model_dir, len(y), spec.val_loss)
        return result
