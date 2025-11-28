"""Plotting helpers for Layer A artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_pred_vs_actual(pred_df: pd.DataFrame, out_path: str) -> None:
    """Line plot of predicted vs actual simple returns over time."""
    if pred_df.empty:
        return
    df = pred_df.sort_values("timestamp")
    df = df.copy()
    df["y_true_ret"] = np.exp(df["y_true_logret"]) - 1.0
    df["y_pred_ret"] = np.exp(df["y_pred_logret"]) - 1.0
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["y_true_ret"], label="actual", linewidth=1.2)
    plt.plot(df["timestamp"], df["y_pred_ret"], label="predicted", linewidth=1.0, alpha=0.8)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Pred vs Actual (simple return)")
    plt.xlabel("Time")
    plt.ylabel("Simple return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter(pred_df: pd.DataFrame, out_path: str) -> None:
    """Scatter plot of predicted vs actual simple returns with 45° line."""
    if pred_df.empty:
        return
    df = pred_df.copy()
    df["y_true_ret"] = np.exp(df["y_true_logret"]) - 1.0
    df["y_pred_ret"] = np.exp(df["y_pred_logret"]) - 1.0
    df = df.dropna(subset=["y_true_ret", "y_pred_ret"])
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(df["y_true_ret"], df["y_pred_ret"], s=12, alpha=0.6)
    lim = _symmetric_lim(df[["y_true_ret", "y_pred_ret"]])
    plt.plot(lim, lim, color="red", linestyle="--", linewidth=1.0, label="45°")
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("Actual simple return")
    plt.ylabel("Predicted simple return")
    plt.title("Pred vs Actual Scatter (simple return)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_train_curves(history_path: str, out_path: str) -> None:
    """Plot train/val loss curves from a history.json file."""
    if not os.path.exists(history_path):
        return
    with open(history_path, "r") as f:
        hist = json.load(f)
    train_loss = hist.get("train_loss", [])
    val_loss = hist.get("val_loss", [])
    if not train_loss and not val_loss:
        return
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(6, 4))
    if train_loss:
        plt.plot(epochs, train_loss, label="train_loss", linewidth=1.2)
    if val_loss:
        plt.plot(epochs, val_loss, label="val_loss", linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _symmetric_lim(df: pd.DataFrame, pad: float = 0.0) -> tuple[float, float]:
    """Symmetric axis limits around zero covering both columns."""
    vals = df.to_numpy().ravel()
    finite = vals[~pd.isna(vals)]
    if finite.size == 0:
        return (-1.0, 1.0)
    m = finite.max()
    mi = finite.min()
    bound = max(abs(m), abs(mi)) + pad
    if bound == 0:
        bound = 1.0
    return (-bound, bound)
