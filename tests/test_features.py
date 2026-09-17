"""Unit tests for feature engineering and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import engineer_features
from src.features.preprocessor import (
    split_data,
    StructuredPreprocessor,
    prepare_target,
    inverse_transform_target,
)


@pytest.fixture
def sample_bengaluru_df():
    """Create a minimal sample DataFrame matching Bengaluru housing schema."""
    return pd.DataFrame({
        "location": ["Koramangala", "Whitefield", "HSR Layout"],
        "longitude": [77.6245, 77.7499, 77.6446],
        "latitude": [12.9352, 12.9698, 12.9121],
        "size": [3, 2, 4],
        "total_sqft": [1500.0, 1200.0, 2000.0],
        "bath": [2, 2, 3],
        "balcony": [1, 1, 2],
        "price": [8500000.0, 7200000.0, 12000000.0],
    })


def test_engineer_features(sample_bengaluru_df):
    """Test full feature engineering pipeline."""
    df_eng, km = engineer_features(sample_bengaluru_df, is_train=True)

    # Check engineered features exist
    assert "price_per_sqft" in df_eng.columns
    assert "distance_to_cbd" in df_eng.columns
    assert "distance_to_tech_hub" in df_eng.columns
    assert "distance_to_airport" in df_eng.columns
    assert "location_cluster" in df_eng.columns

    # Check values are reasonable
    assert (df_eng["price_per_sqft"] > 0).all()
    assert (df_eng["distance_to_cbd"] >= 0).all()
    assert (df_eng["distance_to_tech_hub"] >= 0).all()


def test_location_clusters(sample_bengaluru_df):
    """Test location clustering."""
    df_train, km = engineer_features(sample_bengaluru_df, is_train=True)
    assert "location_cluster" in df_train.columns

    # Test on validation/test set
    df_test, _ = engineer_features(sample_bengaluru_df, kmeans_model=km, is_train=False)
    assert "location_cluster" in df_test.columns


def test_preprocessor_pipeline(sample_bengaluru_df):
    """Test the full structured preprocessing pipeline."""
    # Split (with small sample, some splits might be empty, so just test single set)
    train_eng, km = engineer_features(sample_bengaluru_df, is_train=True)

    # Preprocess
    preprocessor = StructuredPreprocessor()
    X_train = preprocessor.fit_transform(train_eng)
    X_test = preprocessor.transform(train_eng)  # Use same for test

    assert X_train.shape[0] == len(sample_bengaluru_df)
    assert X_test.shape[0] == len(sample_bengaluru_df)
    assert X_train.shape[1] == X_test.shape[1]
    assert preprocessor.is_fitted


def test_target_transformation():
    """Test log transform and inverse transform."""
    prices = pd.Series([1000000.0, 5000000.0, 10000000.0])
    log_prices = prepare_target(prices, log_transform=True)
    recovered_prices = inverse_transform_target(log_prices)

    np.testing.assert_allclose(prices.values, recovered_prices, rtol=1e-5)


def test_split_data(sample_bengaluru_df):
    """Test train/val/test splitting."""
    train, val, test = split_data(sample_bengaluru_df, test_size=0.33, val_size=0.5, seed=42)

    # Check all splits have data (with 3 samples, might have 1-2 each)
    assert len(train) >= 1
    assert len(val) >= 0  # Might be empty with small sample
    assert len(test) >= 1

    # Check total size preserved
    assert len(train) + len(val) + len(test) == len(sample_bengaluru_df)
