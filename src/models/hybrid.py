"""
Hybrid models that combine structured features with NLP embeddings.

The key architectural idea: concatenate scaled structured features
with PCA-reduced text embeddings to create a fused feature matrix,
then train the same model families on this richer representation.

Design decisions:
- Simple concatenation is the fusion strategy. More complex approaches
  (attention-based fusion, cross-modal transformers) are overkill for
  our dataset size and add complexity without proportional gain.
- PCA reduction of embeddings (768 → 50) ensures text features don't
  numerically dominate the ~20 structured features.
- We retrain ElasticNet and Random Forest on fused features to get
  a clean apples-to-apples comparison with structured-only baselines.

Interview talking points:
- "Feature concatenation is a simple but effective early fusion strategy.
   Late fusion (separate models per modality, combined predictions) is
   an alternative but adds complexity and needs more data to justify."
- "PCA reduction serves two purposes: dimensionality reduction to prevent
   overfitting, and balancing the feature count between modalities."
"""

from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from src.config import AppConfig
from src.utils.logger import get_logger
from src.utils.io import save_pickle

logger = get_logger(__name__)


def fuse_features(
    X_structured: np.ndarray,
    X_text: np.ndarray,
) -> np.ndarray:
    """
    Combine structured features and text embeddings via concatenation.

    Args:
        X_structured: (N, d_struct) structured feature matrix.
        X_text: (N, d_text) text embedding matrix (PCA-reduced).

    Returns:
        (N, d_struct + d_text) fused feature matrix.
    """
    if X_structured.shape[0] != X_text.shape[0]:
        raise ValueError(
            f"Sample count mismatch: structured={X_structured.shape[0]}, "
            f"text={X_text.shape[0]}"
        )

    fused = np.hstack([X_structured, X_text])
    logger.info(
        f"Fused features: structured {X_structured.shape[1]} + "
        f"text {X_text.shape[1]} = {fused.shape[1]} total"
    )
    return fused


def build_hybrid_models(config=None) -> Dict[str, object]:
    """
    Create hybrid model instances.

    Same model families as baselines, retrained on fused features.
    """
    config = config or AppConfig.model

    models = {
        "hybrid_elasticnet": ElasticNet(
            alpha=config.elasticnet_alpha,
            l1_ratio=config.elasticnet_l1_ratio,
            random_state=config.random_seed,
            max_iter=5000,
        ),
        "hybrid_random_forest": RandomForestRegressor(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=config.random_seed,
            n_jobs=-1,
        ),
    }

    logger.info(f"Built {len(models)} hybrid model instances")
    return models


def train_hybrid_models(
    X_fused: np.ndarray,
    y_train: np.ndarray,
    save_dir=None,
) -> Dict[str, object]:
    """
    Train all hybrid models on fused features.

    Args:
        X_fused: Fused feature matrix (structured + text embeddings).
        y_train: Training target array.
        save_dir: Directory to save models.

    Returns:
        Dictionary of model_name → fitted model.
    """
    save_dir = save_dir or AppConfig.paths.models
    models = build_hybrid_models()
    fitted = {}

    for name, model in models.items():
        logger.info(f"Training {name} on {X_fused.shape[0]} samples, "
                     f"{X_fused.shape[1]} features")
        model.fit(X_fused, y_train)
        fitted[name] = model

        save_path = save_dir / f"{name}.pkl"
        save_pickle(model, save_path)

        # Log info
        if hasattr(model, "coef_"):
            n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
            logger.info(f"  {name}: {n_nonzero}/{len(model.coef_)} non-zero coefficients")

    logger.info(f"All hybrid models trained and saved to {save_dir}")
    return fitted
