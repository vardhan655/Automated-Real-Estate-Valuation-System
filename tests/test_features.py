"""Unit tests for feature engineering and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    add_ratio_features,
    add_distance_features,
    add_location_clusters,
    engineer_features,
)
from src.features.preprocessor import (
    split_data,
    StructuredPreprocessor,
    prepare_target,
    inverse_transform_target,
)


@pytest.fixture
def sample_df():
    """Create a minimal sample DataFrame matching California housing schema."""
    return pd.DataFrame({
        "longitude": [-122.23, -122.22, -122.24],
        "latitude": [37.88, 37.86, 37.85],
        "housing_median_age": [41.0, 21.0, 52.0],
        "total_rooms": [880.0, 7099.0, 1467.0],
        "total_bedrooms": [129.0, 1106.0, 190.0],
        "population": [322.0, 2401.0, 496.0],
        "households": [126.0, 1138.0, 177.0],
        "median_income": [8.3252, 8.3014, 7.2574],
        "median_house_value": [452600.0, 358500.0, 352100.0],
    })


def test_ratio_features(sample_df):
    """Test ratio feature creation."""
    df = add_ratio_features(sample_df)
    assert "rooms_per_household" in df.columns
    assert "bedrooms_per_room" in df.columns
    assert "population_per_household" in df.columns
    assert (df["rooms_per_household"] > 0).all()


def test_distance_features(sample_df):
    """Test distance feature calculation."""
    df = add_distance_features(sample_df)
    assert "distance_to_sf" in df.columns
    assert "distance_to_la" in df.columns
    assert "distance_to_coast" in df.columns
    # San Francisco points should be very close to SF
    assert (df["distance_to_sf"] < 50).all()


def test_location_clusters(sample_df):
    """Test location clustering."""
    df, km = add_location_clusters(sample_df, n_clusters=2, is_train=True)
    assert "location_cluster" in df.columns
    assert df["location_cluster"].nunique() <= 2

    # Test test-set application
    df_test, _ = add_location_clusters(sample_df, kmeans_model=km, is_train=False)
    assert "location_cluster" in df_test.columns


def test_preprocessor_pipeline(sample_df):
    """Test the full structured preprocessing pipeline."""
    # Split
    train, val, test = split_data(sample_df, test_size=0.33, val_size=0.5, seed=42)

    # Engineer
    train_eng, km = engineer_features(train, is_train=True)
    val_eng, _ = engineer_features(val, kmeans_model=km, is_train=False)

    # Preprocess
    preprocessor = StructuredPreprocessor()
    X_train = preprocessor.fit_transform(train_eng)
    X_val = preprocessor.transform(val_eng)

    assert X_train.shape[0] == len(train)
    assert X_val.shape[0] == len(val)
    assert X_train.shape[1] == X_val.shape[1]
    assert preprocessor.is_fitted


def test_target_transformation():
    """Test log transform and inverse transform."""
    prices = pd.Series([100000.0, 250000.0, 500000.0])
    log_prices = prepare_target(prices, log_transform=True)
    recovered_prices = inverse_transform_target(log_prices)

    np.testing.assert_allclose(prices.values, recovered_prices, rtol=1e-5)
