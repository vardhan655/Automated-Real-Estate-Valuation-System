"""
Preprocessing pipeline for structured features.

Handles:
- Train/val/test splitting with no leakage
- Numeric scaling (StandardScaler)
- Location cluster one-hot encoding
- Target log-transformation (house prices are right-skewed)
- Assembling the final structured feature matrix

Design decisions:
- StandardScaler over MinMaxScaler: more robust to outliers,
  expected by ElasticNet's L1/L2 regularization.
- Log-transform the target: house prices are multiplicative not additive,
  so log-space regression is more appropriate. Predictions are
  back-transformed for evaluation.
- One-hot encode location_cluster: it's categorical (nominal), not ordinal.
  With ~30 clusters, this adds 30 features — acceptable for our dataset size.
- All scalers/encoders fitted on train only, applied to val/test.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import AppConfig
from src.utils.logger import get_logger
from src.utils.io import save_pickle, load_pickle

logger = get_logger(__name__)


def split_data(
    df: pd.DataFrame,
    test_size: float = None,
    val_size: float = None,
    seed: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, validation, and test sets.

    Uses stratified-like approach: first split off test, then split
    remaining into train and validation.

    Args:
        df: Full DataFrame.
        test_size: Fraction for test set (default from config).
        val_size: Fraction of remaining for validation (default from config).
        seed: Random seed (default from config).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    config = AppConfig.data
    test_size = test_size or config.test_size
    val_size = val_size or config.val_size
    seed = seed or config.random_seed

    # First split: separate test set
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)

    # Second split: separate validation from training
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)

    logger.info(
        f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)} "
        f"(total: {len(df)})"
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class StructuredPreprocessor:
    """
    Fits and transforms structured features for model input.

    Fitted on training data only. Transform method works on any split.
    """

    def __init__(self, config=None):
        self.config = config or AppConfig
        self.numeric_scaler: Optional[StandardScaler] = None
        self.cluster_encoder: Optional[OneHotEncoder] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False

    def _get_numeric_columns(self) -> List[str]:
        """Get all numeric feature columns (original + engineered, excluding cluster)."""
        return (
            self.config.data.numeric_features
            + [f for f in self.config.data.engineered_features if f != "location_cluster"]
        )

    def fit(self, df: pd.DataFrame) -> "StructuredPreprocessor":
        """
        Fit scalers and encoders on training data.

        Args:
            df: Training DataFrame (after feature engineering).

        Returns:
            self for chaining.
        """
        numeric_cols = self._get_numeric_columns()

        # Fit numeric scaler
        self.numeric_scaler = StandardScaler()
        self.numeric_scaler.fit(df[numeric_cols])
        logger.info(f"Fitted StandardScaler on {len(numeric_cols)} numeric features")

        # Fit cluster encoder
        if "location_cluster" in df.columns:
            self.cluster_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            )
            self.cluster_encoder.fit(df[["location_cluster"]])
            cluster_names = [f"cluster_{int(c)}" for c in self.cluster_encoder.categories_[0]]
            logger.info(f"Fitted OneHotEncoder for {len(cluster_names)} location clusters")
        else:
            cluster_names = []

        # Build complete feature name list
        self.feature_names = numeric_cols + cluster_names

        # Also include raw lat/lon (scaled) since they carry signal
        if "latitude" in df.columns:
            location_cols = self.config.data.location_features
            self.feature_names = location_cols + self.feature_names

        self.is_fitted = True
        logger.info(f"Preprocessor fitted. Total structured features: {len(self.feature_names)}")

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a DataFrame into the structured feature matrix.

        Args:
            df: DataFrame (train, val, or test) after feature engineering.

        Returns:
            2D numpy array of shape (n_samples, n_features).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call .fit() first.")

        numeric_cols = self._get_numeric_columns()
        parts = []

        # Raw lat/lon (already in reasonable scale, but let's include them)
        if "latitude" in df.columns:
            location_cols = self.config.data.location_features
            parts.append(df[location_cols].values)

        # Scaled numeric features
        numeric_scaled = self.numeric_scaler.transform(df[numeric_cols])
        parts.append(numeric_scaled)

        # One-hot encoded clusters
        if self.cluster_encoder is not None and "location_cluster" in df.columns:
            cluster_encoded = self.cluster_encoder.transform(df[["location_cluster"]])
            parts.append(cluster_encoded)

        X = np.hstack(parts)
        logger.info(f"Transformed structured features: {X.shape}")
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df and return transformed result."""
        return self.fit(df).transform(df)

    def save(self, path=None) -> None:
        """Save fitted preprocessor to disk."""
        path = path or (AppConfig.paths.preprocessors / "structured_preprocessor.pkl")
        save_pickle(self, path)

    @classmethod
    def load(cls, path=None) -> "StructuredPreprocessor":
        """Load a fitted preprocessor from disk."""
        path = path or (AppConfig.paths.preprocessors / "structured_preprocessor.pkl")
        return load_pickle(path)


def prepare_target(
    y: pd.Series,
    log_transform: bool = True,
) -> np.ndarray:
    """
    Prepare the target variable.

    Log-transforming house prices is standard practice because:
    1. Price distributions are right-skewed
    2. Percentage errors matter more than absolute errors for prices
    3. Log-space regression naturally models multiplicative relationships

    Args:
        y: Raw target Series.
        log_transform: Whether to apply log transformation.

    Returns:
        Transformed target array.
    """
    y_arr = y.values.astype(np.float64)
    if log_transform:
        y_arr = np.log1p(y_arr)  # log(1 + y) handles edge cases
        logger.info(f"Log-transformed target. Range: [{y_arr.min():.3f}, {y_arr.max():.3f}]")
    return y_arr


def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
    """Convert log-transformed predictions back to dollar values."""
    return np.expm1(y_log)


if __name__ == "__main__":
    from src.data.loader import load_california_housing
    from src.features.engineer import engineer_features

    df = load_california_housing()
    train, val, test = split_data(df)

    # Engineer features (fit clusters on train only)
    train_eng, km = engineer_features(train, is_train=True)
    val_eng, _ = engineer_features(val, kmeans_model=km, is_train=False)
    test_eng, _ = engineer_features(test, kmeans_model=km, is_train=False)

    # Preprocess
    preprocessor = StructuredPreprocessor()
    X_train = preprocessor.fit_transform(train_eng)
    X_val = preprocessor.transform(val_eng)
    X_test = preprocessor.transform(test_eng)

    y_train = prepare_target(train_eng["median_house_value"])

    print(f"\nStructured feature matrix shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Target (train): {y_train.shape}")
    print(f"\nFeature names ({len(preprocessor.feature_names)}):")
    for i, name in enumerate(preprocessor.feature_names):
        print(f"  {i:3d}: {name}")
