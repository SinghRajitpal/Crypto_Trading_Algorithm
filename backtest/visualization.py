import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_equity(equity: List[float], drawdowns: Optional[List[float]], path: str) -> None:
    if not equity:
        return
    plt.figure(figsize=(9, 4.5))
    plt.plot(equity, label="Equity", color="#1f77b4", linewidth=1.4)
    if drawdowns:
        ax2 = plt.twinx()
        ax2.plot(drawdowns, color="#d62728", alpha=0.4, label="Drawdown")
        ax2.set_ylabel("Drawdown (fraction)", fontsize=11)
        ax2.legend(loc="upper right")
    plt.title("Equity Curve", fontsize=12)
    plt.xlabel("Bars", fontsize=11)
    plt.ylabel("Equity (USD)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_risk_diagnostics(risk_diag: Dict[str, float], path: str) -> None:
    if not risk_diag:
        return
    keys = list(risk_diag.keys())
    vals = [risk_diag[k] for k in keys]
    plt.figure(figsize=(9, 4.5))
    plt.bar(keys, vals, color="#9467bd", alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.title("Risk Diagnostics", fontsize=12)
    plt.ylabel("Value", fontsize=11)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_series(series: Dict[str, List[float]], path: str, title: str) -> None:
    if not series:
        return
    plt.figure(figsize=(9, 4.5))
    for name, vals in series.items():
        plt.plot(vals, label=name, linewidth=1.2)
    plt.title(title, fontsize=12)
    plt.xlabel("Bars", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(x: List[float], y: List[float], path: str, xlabel: str, ylabel: str, title: str) -> None:
    if not x or not y:
        return
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(x, y, alpha=0.6, color="#ff7f0e")
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_summary_json(summary: Dict[str, float], path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def plot_equity_comparison(
    strategy_equity: List[float],
    benchmark_equity: List[float],
    path: str,
) -> None:
    """Plot strategy vs benchmark equity on the same axes."""
    if not strategy_equity or not benchmark_equity:
        return
    # Align lengths to avoid length mismatch during plotting
    n = min(len(strategy_equity), len(benchmark_equity))
    if n == 0:
        return
    plt.figure(figsize=(9, 4.5))
    plt.plot(strategy_equity[:n], label="Strategy", color="#1f77b4", linewidth=1.4)
    plt.plot(benchmark_equity[:n], label="Benchmark", color="#2ca02c", linewidth=1.2, alpha=0.9)
    plt.title("Strategy vs Benchmark Equity", fontsize=12)
    plt.xlabel("Bars", fontsize=11)
    plt.ylabel("Equity (USD)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
