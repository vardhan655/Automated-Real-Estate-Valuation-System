"""
Baseline regression models using only structured features.

Models included:
1. Linear Regression — vanilla baseline, no regularization
2. Ridge — L2 regularization, handles collinearity
3. ElasticNet — L1 + L2, encourages sparsity, good for interpreting
4. Random Forest — non-linear, captures interactions, ensemble robustness

Design decisions:
- All models work in log-price space (target is log-transformed).
- ElasticNet is the signature model for this project: it combines
  regularization with sparsity, letting us identify important features.
- Random Forest captures non-linear location-price interactions that
  linear models miss.
- We train all baselines on structured features only, then compare with
  hybrid (structured + NLP) models to measure NLP contribution.

Interview talking points:
- "ElasticNet's L1 component zeros out irrelevant features, giving us
   automatic feature selection. The L2 component handles correlated
   features like our distance metrics."
- "Random Forest doesn't need feature scaling, but we scale anyway
   for consistency across the pipeline and fair ElasticNet comparison."
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from src.config import AppConfig
from src.utils.logger import get_logger
from src.utils.io import save_pickle

logger = get_logger(__name__)


def build_baseline_models(config=None) -> Dict[str, object]:
    """
    Create all baseline model instances with configured hyperparameters.

    Returns:
        Dictionary of model_name → unfitted sklearn estimator.
    """
    config = config or AppConfig.model

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(
            alpha=config.ridge_alpha,
            random_state=config.random_seed,
        ),
        "elasticnet": ElasticNet(
            alpha=config.elasticnet_alpha,
            l1_ratio=config.elasticnet_l1_ratio,
            random_state=config.random_seed,
            max_iter=5000,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=config.random_seed,
            n_jobs=-1,
        ),
    }

    logger.info(f"Built {len(models)} baseline model instances")
    return models


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "model",
) -> object:
    """
    Train a single model and log basic info.

    Args:
        model: Unfitted sklearn estimator.
        X_train: Training feature matrix.
        y_train: Training target array.
        model_name: Name for logging.

    Returns:
        Fitted model.
    """
    logger.info(f"Training {model_name} on {X_train.shape[0]} samples, "
                f"{X_train.shape[1]} features")
    model.fit(X_train, y_train)
    logger.info(f"  {model_name} trained")

    # Log model-specific info
    if hasattr(model, "coef_"):
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
        logger.info(f"  Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")

    if hasattr(model, "feature_importances_"):
        top_idx = np.argsort(model.feature_importances_)[-5:][::-1]
        logger.info(f"  Top-5 feature importance indices: {top_idx}")

    return model


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_dir=None,
) -> Dict[str, object]:
    """
    Train all baseline models and optionally save to disk.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        save_dir: Directory to save models (default from config).

    Returns:
        Dictionary of model_name → fitted model.
    """
    save_dir = save_dir or AppConfig.paths.models
    models = build_baseline_models()
    fitted = {}

    for name, model in models.items():
        fitted_model = train_model(model, X_train, y_train, model_name=name)
        fitted[name] = fitted_model

        save_path = save_dir / f"baseline_{name}.pkl"
        save_pickle(fitted_model, save_path)

    logger.info(f"All baselines trained and saved to {save_dir}")
    return fitted
