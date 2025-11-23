import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_equity(equity: List[float], drawdowns: Optional[List[float]], path: str) -> None:
    if not equity:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(equity, label="Equity")
    if drawdowns:
        plt.twinx()
        plt.plot(drawdowns, color="red", alpha=0.3, label="Drawdown")
    plt.title("Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_risk_diagnostics(risk_diag: Dict[str, float], path: str) -> None:
    if not risk_diag:
        return
    keys = list(risk_diag.keys())
    vals = [risk_diag[k] for k in keys]
    plt.figure(figsize=(8, 4))
    plt.bar(keys, vals)
    plt.xticks(rotation=45, ha="right")
    plt.title("Risk Diagnostics")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_series(series: Dict[str, List[float]], path: str, title: str) -> None:
    if not series:
        return
    plt.figure(figsize=(8, 4))
    for name, vals in series.items():
        plt.plot(vals, label=name)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(x: List[float], y: List[float], path: str, xlabel: str, ylabel: str, title: str) -> None:
    if not x or not y:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_summary_json(summary: Dict[str, float], path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
