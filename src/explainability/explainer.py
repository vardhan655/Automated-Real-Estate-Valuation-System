"""
Explainability module using SHAP.

Provides global and local explanations for the hybrid model predictions.

Design decisions:
- SHAP (SHapley Additive exPlanations) chosen because:
  1. Model-agnostic: works with ElasticNet, RF, and MLP
  2. Theoretically grounded in cooperative game theory
  3. Provides both global importance and local instance explanations
  4. Industry-standard and interviewers expect it
- For Random Forest: use TreeExplainer (exact, fast)
- For ElasticNet: use LinearExplainer (exact for linear models)
- For fused features: SHAP values cover both structured and text dimensions,
  but text PCA components aren't individually interpretable — we aggregate
  them as "text_embedding_contribution" for honest explanation.

Explainability limitations (state in interviews):
- PCA-reduced text embedding features (text_pca_0, text_pca_1, ...) don't
  have intuitive names. We report their aggregate contribution.
- SHAP background dataset size affects accuracy. We use 200 samples
  as a practical tradeoff.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import AppConfig
from src.utils.logger import get_logger
from src.utils.io import save_pickle, load_pickle

logger = get_logger(__name__)


class ModelExplainer:
    """
    Generates SHAP-based explanations for model predictions.
    """

    def __init__(self, feature_names: List[str], n_text_features: int = 0):
        """
        Args:
            feature_names: List of all feature names (structured + text PCA).
            n_text_features: Number of features that are PCA text embeddings.
                             Used to aggregate text contribution.
        """
        self.feature_names = feature_names
        self.n_text_features = n_text_features
        self.n_structured_features = len(feature_names) - n_text_features
        self.explainer = None
        self.shap_values = None

    def fit(
        self,
        model,
        X_background: np.ndarray,
        model_type: str = "tree",
    ) -> "ModelExplainer":
        """
        Create the SHAP explainer.

        Args:
            model: Fitted sklearn model.
            X_background: Background dataset for SHAP (sample of training data).
            model_type: "tree" for RF, "linear" for ElasticNet, "kernel" for MLP.

        Returns:
            self for chaining.
        """
        max_samples = AppConfig.explainability.shap_max_samples
        if len(X_background) > max_samples:
            idx = np.random.RandomState(42).choice(len(X_background), max_samples, replace=False)
            X_background = X_background[idx]

        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
            logger.info("Created TreeExplainer for Random Forest")
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(model, X_background)
            logger.info("Created LinearExplainer for linear model")
        else:
            self.explainer = shap.KernelExplainer(model.predict, X_background)
            logger.info("Created KernelExplainer (model-agnostic)")

        return self

    def explain_global(
        self,
        X: np.ndarray,
        save_path: Path = None,
    ) -> Dict[str, float]:
        """
        Compute global feature importance from SHAP values.

        Args:
            X: Feature matrix to explain (use validation set).
            save_path: Path to save the importance plot.

        Returns:
            Dict of feature_name → mean absolute SHAP value.
        """
        logger.info(f"Computing SHAP values for {X.shape[0]} samples...")
        self.shap_values = self.explainer.shap_values(X)

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        importance = {}
        for i, name in enumerate(self.feature_names):
            importance[name] = float(mean_abs_shap[i])

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # Log top features
        logger.info("Global feature importance (top 10):")
        for i, (name, val) in enumerate(importance.items()):
            if i >= 10:
                break
            logger.info(f"  {i+1}. {name}: {val:.4f}")

        # Aggregate text contribution
        if self.n_text_features > 0:
            text_contribution = sum(
                v for k, v in importance.items()
                if k.startswith("text_pca_")
            )
            struct_contribution = sum(v for k, v in importance.items()) - text_contribution
            logger.info(f"\n  Text embedding total contribution: {text_contribution:.4f}")
            logger.info(f"  Structured features total contribution: {struct_contribution:.4f}")
            logger.info(f"  Text share: {text_contribution/(text_contribution+struct_contribution):.1%}")

        # Plot
        if save_path:
            self._plot_global_importance(importance, save_path)

        return importance

    def explain_local(
        self,
        X_instance: np.ndarray,
        instance_idx: int = 0,
    ) -> Dict[str, float]:
        """
        Explain a single prediction.

        Args:
            X_instance: (1, D) or (D,) feature vector.
            instance_idx: For logging purposes.

        Returns:
            Dict of feature_name → SHAP value for this instance.
        """
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)

        shap_vals = self.explainer.shap_values(X_instance)[0]

        explanation = {}
        for i, name in enumerate(self.feature_names):
            explanation[name] = float(shap_vals[i])

        # Sort by absolute impact
        explanation = dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True))

        logger.info(f"Local explanation for instance {instance_idx}:")
        for i, (name, val) in enumerate(explanation.items()):
            if i >= 8:
                break
            direction = "↑" if val > 0 else "↓"
            logger.info(f"  {name}: {val:+.4f} {direction}")

        # Aggregate text contribution for this instance
        if self.n_text_features > 0:
            text_effect = sum(v for k, v in explanation.items() if k.startswith("text_pca_"))
            direction = "increases" if text_effect > 0 else "decreases"
            logger.info(f"  Text embedding net effect: {text_effect:+.4f} ({direction} predicted price)")

        return explanation

    def get_text_contribution_summary(self) -> Dict[str, float]:
        """
        Summarize the aggregate contribution of text features vs structured.

        This is the honest way to explain NLP contribution when features
        are PCA components without semantic names.
        """
        if self.shap_values is None:
            raise RuntimeError("Call explain_global first to compute SHAP values")

        mean_abs = np.abs(self.shap_values).mean(axis=0)

        structured_importance = float(mean_abs[:self.n_structured_features].sum())
        text_importance = float(mean_abs[self.n_structured_features:].sum())
        total = structured_importance + text_importance

        return {
            "structured_total": structured_importance,
            "text_total": text_importance,
            "text_share_pct": round(text_importance / total * 100, 2) if total > 0 else 0,
            "structured_share_pct": round(structured_importance / total * 100, 2) if total > 0 else 0,
        }

    def _plot_global_importance(self, importance: Dict, save_path: Path) -> None:
        """Plot horizontal bar chart of feature importance."""
        top_n = AppConfig.explainability.top_features

        names = list(importance.keys())[:top_n]
        values = list(importance.values())[:top_n]

        # Reverse for horizontal bar (top feature at top)
        names = names[::-1]
        values = values[::-1]

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ["#e74c3c" if "text_pca" in n else "#3498db" for n in names]
        ax.barh(range(len(names)), values, color=colors, edgecolor="white")

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title("Global Feature Importance (SHAP)", fontsize=14)

        # Legend for color coding
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#3498db", label="Structured features"),
            Patch(facecolor="#e74c3c", label="Text embedding (PCA)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved SHAP importance plot → {save_path}")
        plt.close()
