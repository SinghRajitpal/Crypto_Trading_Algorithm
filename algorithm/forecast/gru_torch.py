"""PyTorch GRU forecaster for next-bar log returns."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SequenceStandardizer:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "SequenceStandardizer":
        feat_mean = np.nanmean(X, axis=(0, 1))
        feat_std = np.nanstd(X, axis=(0, 1))
        feat_std[feat_std == 0] = 1.0
        tgt_mean = float(np.nanmean(y))
        tgt_std = float(np.nanstd(y))
        if tgt_std == 0:
            tgt_std = 1.0
        return cls(feat_mean, feat_std, tgt_mean, tgt_std)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean) / self.feature_std

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def inverse_target(self, y_std: np.ndarray) -> np.ndarray:
        return (y_std * self.target_std) + self.target_mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SequenceStandardizer":
        return cls(
            feature_mean=np.array(payload["feature_mean"], dtype=float),
            feature_std=np.array(payload["feature_std"], dtype=float),
            target_mean=float(payload["target_mean"]),
            target_std=float(payload["target_std"]),
        )


@dataclass
class GRUSpec:
    timeframe: str
    lookback: int
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    huber_delta: float
    grad_clip: float
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    val_loss: Optional[float] = None

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "GRUSpec":
        with open(path, "r") as f:
            data = json.load(f)
        if "weight_decay" not in data:
            data["weight_decay"] = 0.0
        return cls(**data)


class _GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class GRUForecasterTorch:
    def __init__(
        self,
        lookback: int = config.GRU_LOOKBACK,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        huber_delta: float = 1.0,
        grad_clip: float = 1.0,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 5,
        validation_split: float = 0.2,
        device: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        self.lookback = lookback
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.huber_delta = huber_delta
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.verbose = verbose
        self.device = self._resolve_device(device)

        self.model: Optional[_GRUModel] = None
        self.scaler: Optional[SequenceStandardizer] = None

    @staticmethod
    def _resolve_device(preferred: Optional[str]) -> torch.device:
        if preferred and preferred.lower() == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred and preferred.lower() == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, list], GRUSpec]:
        if X.ndim != 3 or y.ndim != 1:
            raise ValueError("Expected X shape (n_samples, seq_len, n_features) and y shape (n_samples,)")
        if X.shape[1] != self.lookback or X.shape[2] != self.input_size:
            raise ValueError(f"Expected X shape (*, {self.lookback}, {self.input_size}), got {X.shape}")

        # Train/val split on raw features; scaler uses train only to avoid leakage
        n = len(y)
        val_size = max(1, int(n * self.validation_split))
        train_size = n - val_size
        X_train_raw, X_val_raw = X[:train_size], X[train_size:]
        y_train_raw, y_val_raw = y[:train_size], y[train_size:]

        self.scaler = SequenceStandardizer.fit(X_train_raw, y_train_raw)
        X_train = self.scaler.transform_features(X_train_raw)
        y_train = self.scaler.transform_target(y_train_raw)
        X_val = self.scaler.transform_features(X_val_raw)
        y_val = self.scaler.transform_target(y_val_raw)

        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.model = _GRUModel(self.input_size, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.SmoothL1Loss(beta=self.huber_delta)

        best_state = None
        best_val = float("inf")
        no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.verbose:
                logger.info("Epoch %d/%d | train_loss=%.6f | val_loss=%.6f", epoch, self.epochs, train_loss, val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        spec = GRUSpec(
            timeframe=config.GRU_TIMEFRAME,
            lookback=self.lookback,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            epochs=len(history["val_loss"]),
            huber_delta=self.huber_delta,
            grad_clip=self.grad_clip,
            val_loss=best_val,
        )
        return history, spec

    def predict_log_return(self, window: np.ndarray) -> float:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained or loaded.")
        if window.shape != (self.lookback, self.input_size):
            raise ValueError(f"Expected window shape ({self.lookback}, {self.input_size}), got {window.shape}")
        window_std = self.scaler.transform_features(window[np.newaxis, ...])
        tensor = torch.from_numpy(window_std).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_std = self.model(tensor).cpu().numpy()[0]
        return float(self.scaler.inverse_target(np.array([pred_std]))[0])

    def predict_batch(self, windows: np.ndarray, return_std: bool = False) -> np.ndarray:
        """Predict log returns for a batch of windows.

        Args:
            windows: ndarray (n_samples, lookback, input_size).
            return_std: if True, return standardized predictions instead of de-standardized.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained or loaded.")
        if windows.ndim != 3 or windows.shape[1:] != (self.lookback, self.input_size):
            raise ValueError(f"Expected windows shape (*, {self.lookback}, {self.input_size}), got {windows.shape}")
        windows_std = self.scaler.transform_features(windows)
        tensor = torch.from_numpy(windows_std).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds_std = self.model(tensor).cpu().numpy()
        if return_std:
            return preds_std
        return self.scaler.inverse_target(preds_std)

    def predict_simple_return(self, window: np.ndarray) -> float:
        log_ret = self.predict_log_return(window)
        return float(np.exp(log_ret) - 1.0)

    def save(self, model_dir: str, spec: GRUSpec) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model/scaler missing; call fit or load first.")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pt")
        scaler_path = os.path.join(model_dir, "scaler.json")
        spec_path = os.path.join(model_dir, "gru_spec.json")

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            model_path,
        )
        with open(scaler_path, "w") as f:
            json.dump(self.scaler.to_dict(), f, indent=2)
        spec.to_json(spec_path)
        logger.info("Saved PyTorch GRU model → %s", model_path)

    @classmethod
    def load(cls, model_dir: str, device: Optional[str] = None) -> "GRUForecasterTorch":
        model_path = os.path.join(model_dir, "model.pt")
        scaler_path = os.path.join(model_dir, "scaler.json")
        spec_path = os.path.join(model_dir, "gru_spec.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model weights: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler file: {scaler_path}")

        with open(scaler_path, "r") as f:
            scaler = SequenceStandardizer.from_dict(json.load(f))
        spec = GRUSpec.from_json(spec_path) if os.path.exists(spec_path) else None
        state = torch.load(model_path, map_location="cpu")
        forecaster = cls(
            lookback=getattr(spec, "lookback", config.GRU_LOOKBACK) if spec else config.GRU_LOOKBACK,
            input_size=state.get("input_size", scaler.feature_mean.shape[-1]),
            hidden_size=state.get("hidden_size", 64),
            num_layers=state.get("num_layers", 1),
            dropout=state.get("dropout", 0.0),
            learning_rate=getattr(spec, "learning_rate", 1e-3) if spec else 1e-3,
            weight_decay=getattr(spec, "weight_decay", 0.0) if spec else 0.0,
            huber_delta=getattr(spec, "huber_delta", 1.0) if spec else 1.0,
            grad_clip=getattr(spec, "grad_clip", 1.0) if spec else 1.0,
            batch_size=getattr(spec, "batch_size", 32) if spec else 32,
            epochs=getattr(spec, "epochs", 1) if spec else 1,
            device=device or config.GRU_DEVICE,
        )
        forecaster.scaler = scaler
        forecaster.model = _GRUModel(
            forecaster.input_size,
            forecaster.hidden_size,
            forecaster.num_layers,
            forecaster.dropout,
        ).to(forecaster.device)
        forecaster.model.load_state_dict(state["state_dict"])
        forecaster.model.eval()
        return forecaster
