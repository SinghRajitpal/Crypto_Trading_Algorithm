"""Layer A walk-forward trainer for GRU-based forecaster."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import config
from algorithm.forecast.gru_torch import GRUForecasterTorch, GRUSpec
from data.gru_sequence_builder import build_sequences_with_index
from data.historical_data import HistoricalDataFetcher
from backtest.plot_layer_a import plot_pred_vs_actual, plot_scatter, plot_train_curves
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GRULayerResult:
    """Artifacts produced by Layer A walk-forward training/inference."""

    symbol: str
    timeframe: str
    lookback: int
    retrain_days: int
    artifact_dir: str
    predictions_path: str
    metrics_path: str
    summary_path: str
    samples: int
    train_start: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "GRULayerResult":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    s1 = pd.Series(a).rank()
    s2 = pd.Series(b).rank()
    return _corr(s1.to_numpy(), s2.to_numpy())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics for predictions vs realized."""
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "count": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "ic_pearson": float("nan"),
            "ic_spearman": float("nan"),
            "sign_accuracy": float("nan"),
        }
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    ic_pearson = _corr(y_true, y_pred)
    ic_spearman = _spearman(y_true, y_pred)
    sign_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {
        "count": int(len(y_true)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "ic_pearson": ic_pearson,
        "ic_spearman": ic_spearman,
        "sign_accuracy": sign_accuracy,
    }


class GRULayerTrainer:
    """Walk-forward Layer A trainer that produces saved predictions/metrics/checkpoints."""

    def __init__(
        self,
        symbol: str,
        timeframe: str = config.GRU_TIMEFRAME,
        lookback: int = config.GRU_LOOKBACK,
        retrain_days: int = getattr(config, "GRU_RETRAIN_DAYS", 30),
        min_train_samples: int = getattr(config, "GRU_MIN_TRAIN_SAMPLES", 200),
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.lookback = lookback
        self.retrain_days = retrain_days
        self.min_train_samples = min_train_samples

    async def run(
        self,
        start: datetime,
        end: datetime,
        output_dir: str,
    ) -> GRULayerResult:
        os.makedirs(output_dir, exist_ok=True)
        fetcher = HistoricalDataFetcher(demo=False)
        lookback_start = _lookback_start(start, self.timeframe)
        try:
            try:
                df = await fetcher.download_ohlcv(
                    self.symbol,
                    self.timeframe,
                    lookback_start,
                    end,
                    force=False,
                    cache_only=True,
                )
            except FileNotFoundError:
                logger.info("Cache miss for %s %s; attempting network fetch.", self.symbol, self.timeframe)
                df = await fetcher.download_ohlcv(
                    self.symbol,
                    self.timeframe,
                    lookback_start,
                    end,
                    force=False,
                    cache_only=False,
                )
        finally:
            # do not close yet; funding fetch below uses the same client
            pass

        if df is None or df.empty:
            raise ValueError("No historical data available for GRU training.")

        df = df.sort_index()
        try:
            funding_start = df.index.min()
            funding_series = await fetcher.fetch_funding_rate(
                self.symbol,
                start=funding_start,
                end=end,
                force=False,
            )
            funding_series = funding_series.reindex(df.index).fillna(0.0)
        except Exception:
            funding_series = None
        finally:
            await fetcher.close()

        X, y, ts = build_sequences_with_index(
            df,
            lookback=self.lookback,
            funding=funding_series,
        )
        if X.size == 0 or y.size == 0:
            raise ValueError("Insufficient data to build GRU training sequences.")

        # Feature stats logging for diagnostics
        try:
            feat_names = list(config.GRU_FEATURE_SCHEMA)
            flat = X.reshape(-1, X.shape[-1])
            stats = {}
            for i, name in enumerate(feat_names):
                col = flat[:, i]
                stats[name] = {
                    "mean": float(np.nanmean(col)),
                    "std": float(np.nanstd(col)),
                    "min": float(np.nanmin(col)),
                    "max": float(np.nanmax(col)),
                }
            for name, s in stats.items():
                logger.info(
                    "Feature stats | %s | mean=%.4f std=%.4f min=%.4f max=%.4f",
                    name,
                    s["mean"],
                    s["std"],
                    s["min"],
                    s["max"],
                )
                if s["std"] == 0 or np.isnan(s["std"]):
                    logger.warning("Degenerate feature detected (std=0 or NaN): %s", name)
        except Exception as exc:
            logger.warning("Feature stats logging failed: %s", exc)

        artifacts = self._walk_forward(X, y, ts, start, end, output_dir)
        predictions_path = os.path.join(output_dir, "predictions.csv")
        metrics_path = os.path.join(output_dir, "metrics.jsonl")
        summary_path = os.path.join(output_dir, "summary.json")

        artifacts["predictions"].to_csv(predictions_path, index=False)
        with open(metrics_path, "w") as f:
            for rec in artifacts["step_metrics"]:
                f.write(json.dumps(rec) + "\n")
        with open(summary_path, "w") as f:
            json.dump(artifacts["summary"], f, indent=2)

        # Plots
        try:
            self._generate_plots(artifacts, output_dir)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Plot generation failed for %s: %s", self.symbol, exc)

        result = GRULayerResult(
            symbol=self.symbol,
            timeframe=self.timeframe,
            lookback=self.lookback,
            retrain_days=self.retrain_days,
            artifact_dir=output_dir,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            samples=int(len(y)),
            train_start=str(df.index.min()),
            test_start=str(start),
            test_end=str(end),
        )
        result_path = os.path.join(output_dir, "gru_spec.json")
        result.to_json(result_path)
        logger.info(
            "GRU Layer A complete | symbol=%s | preds=%d | steps=%d | output=%s",
            self.symbol,
            len(artifacts["predictions"]),
            len(artifacts["step_metrics"]),
            output_dir,
        )
        return result

    def _walk_forward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ts: np.ndarray,
        test_start: datetime,
        test_end: datetime,
        output_dir: str,
    ) -> Dict[str, Any]:
        retrain_delta = timedelta(days=self.retrain_days)
        anchor = pd.Timestamp(test_start)
        test_end_ts = pd.Timestamp(test_end)
        step = 0
        preds: List[Dict[str, Any]] = []
        step_metrics: List[Dict[str, Any]] = []
        checkpoints_root = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoints_root, exist_ok=True)
        ts_index = pd.to_datetime(ts)

        while anchor < test_end_ts:
            train_mask = ts_index < anchor
            if np.sum(train_mask) < self.min_train_samples:
                logger.warning(
                    "Stopping walk-forward early (insufficient train samples: %d < %d)",
                    np.sum(train_mask),
                    self.min_train_samples,
                )
                break
            X_train, y_train, ts_train = X[train_mask], y[train_mask], ts_index[train_mask]
            forecaster = self._init_forecaster()
            history, spec = forecaster.fit(X_train, y_train)
            spec.train_start = str(pd.to_datetime(ts_train.min()))
            spec.train_end = str(pd.to_datetime(ts_train.max()))
            if history.get("val_loss"):
                spec.val_loss = float(history["val_loss"][-1])
                spec.epochs = len(history["val_loss"])

            model_dir = os.path.join(checkpoints_root, f"step_{step:03d}")
            forecaster.save(model_dir, spec)
            with open(os.path.join(model_dir, "history.json"), "w") as hf:
                json.dump(history, hf, indent=2)

            pred_start = anchor
            pred_end = min(anchor + retrain_delta, test_end_ts)
            pred_mask = (ts_index >= pred_start) & (ts_index < pred_end)
            if np.any(pred_mask):
                X_pred, y_true, ts_pred = X[pred_mask], y[pred_mask], ts_index[pred_mask]
                y_pred = forecaster.predict_batch(X_pred)
                y_pred_std = forecaster.predict_batch(X_pred, return_std=True)
                metrics = compute_metrics(y_true, y_pred)
                metrics.update(
                    {
                        "symbol": self.symbol,
                        "step": step,
                        "train_samples": int(len(y_train)),
                        "pred_samples": int(len(y_pred)),
                        "train_start": spec.train_start,
                        "train_end": spec.train_end,
                        "pred_start": pred_start.isoformat(),
                        "pred_end": pred_end.isoformat(),
                        "val_loss": spec.val_loss,
                    }
                )
                step_metrics.append(metrics)
                for ts_i, yp, yp_std, yt in zip(ts_pred, y_pred, y_pred_std, y_true):
                    ts_pd = pd.to_datetime(ts_i)
                    preds.append(
                        {
                            "timestamp": ts_pd.isoformat(),
                            "timestamp_ms": int(ts_pd.timestamp() * 1000),
                            "symbol": self.symbol,
                            "y_pred_logret": float(yp),
                            "y_pred_std": float(yp_std),
                            "y_true_logret": float(yt),
                            "step": step,
                            "train_start": spec.train_start,
                            "train_end": spec.train_end,
                            "pred_start": pred_start.isoformat(),
                            "pred_end": pred_end.isoformat(),
                            "model_dir": model_dir,
                        }
                    )

            anchor = pred_end
            step += 1

        pred_df = pd.DataFrame(preds)
        if not pred_df.empty:
            pred_df.sort_values("timestamp_ms", inplace=True)
            summary_metrics = compute_metrics(
                pred_df["y_true_logret"].to_numpy(), pred_df["y_pred_logret"].to_numpy()
            )
        else:
            summary_metrics = compute_metrics(np.array([]), np.array([]))
        summary = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "lookback": self.lookback,
            "retrain_days": self.retrain_days,
            "predictions": len(pred_df),
            "steps": step,
            **summary_metrics,
        }
        return {"predictions": pred_df, "step_metrics": step_metrics, "summary": summary}

    def _init_forecaster(self) -> GRUForecasterTorch:
        return GRUForecasterTorch(
            lookback=self.lookback,
            input_size=len(config.GRU_FEATURE_SCHEMA),
            hidden_size=config.GRU_HIDDEN_SIZE,
            num_layers=config.GRU_NUM_LAYERS,
            dropout=config.GRU_DROPOUT,
            learning_rate=config.GRU_LR,
            weight_decay=0.0,
            optimizer=config.GRU_OPTIMIZER,
            loss=config.GRU_LOSS,
            bidirectional=config.GRU_BIDIRECTIONAL,
            grad_clip=config.GRU_GRAD_CLIP,
            batch_size=config.GRU_BATCH_SIZE,
            epochs=config.GRU_EPOCHS,
            patience=config.GRU_EARLY_STOP_PATIENCE,
            validation_split=config.GRU_VALIDATION_SPLIT,
            verbose=config.GRU_TRAIN_VERBOSE,
            device=config.GRU_DEVICE,
            feature_schema=list(config.GRU_FEATURE_SCHEMA),
        )

    def _generate_plots(self, artifacts: Dict[str, Any], output_dir: str) -> None:
        """Generate plots for Layer A outputs."""
        pred_df: pd.DataFrame = artifacts["predictions"]
        if not pred_df.empty:
            if not np.issubdtype(pred_df["timestamp"].dtype, np.datetime64):
                pred_df = pred_df.copy()
                pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
            plot_pred_vs_actual(pred_df, os.path.join(output_dir, "pred_vs_actual.png"))
            plot_scatter(pred_df, os.path.join(output_dir, "scatter_pred_vs_true.png"))

        # Training curves per step
        checkpoints_root = Path(output_dir) / "checkpoints"
        if checkpoints_root.exists():
            for step_dir in sorted(checkpoints_root.glob("step_*")):
                hist_path = step_dir / "history.json"
                out_path = step_dir.with_name(f"{step_dir.name}_train_curves.png")
                plot_train_curves(str(hist_path), str(out_path))


def _lookback_start(start: datetime, timeframe: str) -> datetime:
    """Compute training lookback start for GRU window with cushion."""
    try:
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
        elif timeframe.endswith("d"):
            hours = int(timeframe[:-1]) * 24
        else:
            hours = 1
        bars_needed = config.GRU_LOOKBACK + 50
        days = max(90, int((bars_needed * hours) / 24) + 30)
    except Exception:
        days = 90
    return start - timedelta(days=days)
