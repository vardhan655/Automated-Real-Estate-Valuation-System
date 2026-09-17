"""
Feature engineering for Bengaluru location-aware and derived features.

This module creates features that add predictive power:
- Derived property features (price per sqft, etc.)
- Bengaluru geographic distance features (distances to MG Road CBD, IT Hubs, BLR Airport)
- Location clustering (K-Means on Bengaluru lat/lon for neighborhood proxying)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import AppConfig
from src.data.bengaluru_localities import (
    BENGALURU_REFERENCE_POINTS,
    get_locality_coordinates,
)
from src.utils.logger import get_logger
from src.utils.io import save_pickle, load_pickle

logger = get_logger(__name__)


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the Haversine distance between two coordinates in kilometers.

    Uses spherical Earth approximation (Radius = 6,371 km).
    """
    R = 6371.0  # Earth radius in kilometers

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))

    return R * c


def add_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived property features.

    Calculates price_per_sqft if price exists, or estimates it.
    """
    df = df.copy()

    # Calculate price_per_sqft if price is present
    if "price" in df.columns:
        df["price_per_sqft"] = df["price"] / df["total_sqft"].clip(lower=100)
    else:
        # Fallback for inference when price isn't known yet
        df["price_per_sqft"] = 5000.0  # default average

    logger.info("Added property features: price_per_sqft")
    return df


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geographic distance features to major Bengaluru hubs:
    - Distance to MG Road / CBD (Central Business District)
    - Distance to Nearest Major Tech Hub (ITPL Whitefield, Electronic City, Manyata, Ecospace)
    - Distance to Kempegowda International Airport (BLR)
    """
    df = df.copy()

    # Coordinates
    cbd_lat, cbd_lon = BENGALURU_REFERENCE_POINTS["cbd"]
    airport_lat, airport_lon = BENGALURU_REFERENCE_POINTS["airport"]

    # Tech hubs
    tech_hubs = [
        BENGALURU_REFERENCE_POINTS["itpl_whitefield"],
        BENGALURU_REFERENCE_POINTS["electronic_city"],
        BENGALURU_REFERENCE_POINTS["manyata_tech_park"],
        BENGALURU_REFERENCE_POINTS["bellandur_ecospace"],
    ]

    # Distance to CBD
    df["distance_to_cbd"] = df.apply(
        lambda r: _haversine_distance(r["latitude"], r["longitude"], cbd_lat, cbd_lon),
        axis=1,
    )

    # Distance to Airport
    df["distance_to_airport"] = df.apply(
        lambda r: _haversine_distance(r["latitude"], r["longitude"], airport_lat, airport_lon),
        axis=1,
    )

    # Distance to nearest Tech Hub
    def min_tech_hub_dist(lat, lon):
        return min(_haversine_distance(lat, lon, th[0], th[1]) for th in tech_hubs)

    df["distance_to_tech_hub"] = df.apply(
        lambda r: min_tech_hub_dist(r["latitude"], r["longitude"]),
        axis=1,
    )

    logger.info("Added distance features: distance_to_cbd, distance_to_tech_hub, distance_to_airport")
    return df


def add_location_clusters(
    df: pd.DataFrame,
    n_clusters: int = None,
    kmeans_model: KMeans = None,
    is_train: bool = True,
) -> tuple:
    """
    Add location cluster labels using K-Means on Bengaluru lat/lon.

    Captures localized neighborhood price variations.
    """
    df = df.copy()
    if n_clusters is None:
        n_clusters = AppConfig.data.n_location_clusters

    coords = df[["latitude", "longitude"]].values

    if is_train:
        # Dynamically adjust n_clusters if dataset is small
        effective_n_clusters = min(n_clusters, len(df))
        kmeans_model = KMeans(
            n_clusters=effective_n_clusters,
            random_state=AppConfig.data.random_seed,
            n_init=10,
        )
        kmeans_model.fit(coords)
        logger.info(f"Fitted location KMeans with {effective_n_clusters} clusters on {len(df)} points")
    else:
        if kmeans_model is None:
            raise ValueError("Must provide a fitted KMeans model for non-training data")
        logger.info(f"Applying pre-fitted location clusters to {len(df)} points")

    df["location_cluster"] = kmeans_model.predict(coords)

    return df, kmeans_model


def engineer_features(
    df: pd.DataFrame,
    kmeans_model: KMeans = None,
    is_train: bool = True,
) -> tuple:
    """
    Run the full feature engineering pipeline for Bengaluru housing.

    Args:
        df: Raw DataFrame with base features (location, size, total_sqft, etc.)
        kmeans_model: Pre-fitted KMeans model (None for training set).
        is_train: Whether this is the training set.

    Returns:
        Tuple of (engineered DataFrame, fitted KMeans model).
    """
    logger.info(f"Engineering features ({'train' if is_train else 'val/test'}, {len(df)} rows)")

    # Ensure latitude and longitude exist
    if "latitude" not in df.columns or "longitude" not in df.columns:
        if "location" in df.columns:
            df = df.copy()
            df["latitude"] = df["location"].apply(lambda loc: get_locality_coordinates(loc)[0])
            df["longitude"] = df["location"].apply(lambda loc: get_locality_coordinates(loc)[1])

    df = add_property_features(df)
    df = add_distance_features(df)
    df, kmeans_model = add_location_clusters(df, kmeans_model=kmeans_model, is_train=is_train)

    n_new_features = len(AppConfig.data.engineered_features)
    logger.info(f"Feature engineering complete. Added {n_new_features} new features. "
                f"Total columns: {len(df.columns)}")

    return df, kmeans_model


if __name__ == "__main__":
    from src.data.bengaluru_loader import load_bengaluru_housing

    df = load_bengaluru_housing()
    df_eng, km = engineer_features(df)

    print(f"\nEngineered columns: {list(df_eng.columns)}")
    print(df_eng[["location", "size", "total_sqft", "distance_to_cbd", "distance_to_tech_hub", "location_cluster"]].head())
