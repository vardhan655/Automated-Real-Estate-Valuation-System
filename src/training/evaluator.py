"""
Model evaluation utilities.

Provides consistent evaluation metrics and visualization across
all model variants (baseline, hybrid, MLP).

Metrics:
- RMSE: Root Mean Squared Error (primary metric, penalizes large errors)
- MAE: Mean Absolute Error (robust to outliers, intuitive "average error")
- R²: Coefficient of determination (proportion of variance explained)
- MAPE: Mean Absolute Percentage Error (relative error, good for prices)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import AppConfig
from src.utils.logger import get_logger
from src.utils.io import save_json

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Both y_true and y_pred should be in the ORIGINAL dollar scale
    (not log-transformed) for interpretable metrics.

    Args:
        y_true: Ground truth prices.
        y_pred: Predicted prices.
        prefix: Optional prefix for metric names (e.g., "val_").

    Returns:
        Dictionary of metric_name → value.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE: careful with near-zero values
    mask = y_true > 1000  # ignore implausible values for MAPE
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    metrics = {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
        f"{prefix}mape": mape,
    }

    logger.info(
        f"  {prefix}RMSE: ${rmse:,.0f} | MAE: ${mae:,.0f} | "
        f"R²: {r2:.4f} | MAPE: {mape:.2f}%"
    )

    return metrics


def compare_models(
    results: Dict[str, Dict[str, float]],
    save_path: Path = None,
) -> None:
    """
    Print a comparison table and save results.

    Args:
        results: Dict of model_name → metrics dict.
        save_path: Path to save comparison JSON.
    """
    print(f"\n{'='*80}")
    print(f"{'Model':<30} {'RMSE ($)':>12} {'MAE ($)':>12} {'R²':>8} {'MAPE (%)':>10}")
    print(f"{'-'*80}")

    for name, metrics in sorted(results.items(), key=lambda x: x[1].get("rmse", x[1].get("val_rmse", 0))):
        rmse = metrics.get("rmse", metrics.get("val_rmse", 0))
        mae = metrics.get("mae", metrics.get("val_mae", 0))
        r2 = metrics.get("r2", metrics.get("val_r2", 0))
        mape = metrics.get("mape", metrics.get("val_mape", 0))
        print(f"  {name:<28} {rmse:>12,.0f} {mae:>12,.0f} {r2:>8.4f} {mape:>10.2f}")

    print(f"{'='*80}")

    if save_path:
        save_json(results, save_path)


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path = None,
) -> None:
    """
    Create a predicted vs actual scatter plot.

    The ideal model would put all points on the diagonal line.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color="#4C72B0")

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect")

    ax.set_xlabel("Actual Price ($)", fontsize=12)
    ax.set_ylabel("Predicted Price ($)", fontsize=12)
    ax.set_title(f"Predicted vs Actual — {model_name}", fontsize=14)
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved prediction plot → {save_path}")
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path = None,
) -> None:
    """
    Create residual distribution and residual vs predicted plots.

    Residual analysis reveals:
    - Heteroscedasticity (error variance changes with price)
    - Systematic bias (model consistently over/under predicts)
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual distribution
    axes[0].hist(residuals, bins=50, color="#4C72B0", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual ($)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"Residual Distribution — {model_name}", fontsize=12)

    # Residuals vs predicted
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=5, color="#4C72B0")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted Price ($)", fontsize=11)
    axes[1].set_ylabel("Residual ($)", fontsize=11)
    axes[1].set_title(f"Residuals vs Predicted — {model_name}", fontsize=12)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved residual plot → {save_path}")
    plt.close()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "rmse",
    save_path: Path = None,
) -> None:
    """Bar chart comparing models on a given metric."""
    names = list(results.keys())
    values = [results[n].get(metric, results[n].get(f"val_{metric}", 0)) for n in names]

    # Sort by performance
    sorted_pairs = sorted(zip(names, values), key=lambda x: x[1])
    names, values = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2ecc71" if "hybrid" in n.lower() else "#3498db" for n in names]
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel(f"{metric.upper()}", fontsize=12)
    ax.set_title(f"Model Comparison — {metric.upper()}", fontsize=14)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}" if metric in ("rmse", "mae") else f"{val:.4f}",
                va="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot → {save_path}")
    plt.close()
