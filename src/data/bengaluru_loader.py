"""
Bengaluru Housing Dataset Loader.

Loads or generates a Bengaluru housing dataset with features:
- location (locality name)
- size (BHK count: 1-5)
- total_sqft (property area in square feet)
- bath (number of bathrooms)
- balcony (number of balconies)
- price (price in ₹ INR)
- latitude, longitude (mapped from locality)

If no real dataset is available, generates a realistic synthetic dataset
based on Bengaluru real estate market patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import AppConfig
from src.data.bengaluru_localities import (
    BENGALURU_LOCALITIES,
    get_locality_coordinates,
)
from src.utils.logger import get_logger
from src.utils.io import save_dataframe

logger = get_logger(__name__)


def generate_synthetic_bengaluru_data(
    n_samples: int = 15000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic Bengaluru housing dataset.

    Pricing model based on:
    - Location (premium vs budget localities)
    - BHK size
    - Total square feet
    - Bathrooms and balconies
    - Realistic noise

    Args:
        n_samples: Number of properties to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with Bengaluru housing data.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Define locality tiers by typical price levels
    premium_localities = [
        "Indiranagar", "Koramangala", "Whitefield", "HSR Layout",
        "Jayanagar", "Malleshwaram", "Frazer Town", "MG Road",
        "Domlur", "Richmond Town", "Sadashiva Nagar"
    ]
    mid_tier_localities = [
        "Marathahalli", "Bellandur", "Sarjapur Road", "Electronic City",
        "Hebbal", "Yelahanka", "JP Nagar", "BTM Layout",
        "Rajaji Nagar", "Vijayanagar", "Kalyan Nagar", "Bannerghatta Road"
    ]
    budget_localities = [
        "Anekal", "Begur", "Hosa Road", "Jigani", "Chandapura",
        "Bommasandra", "Kengeri", "Uttarahalli", "Gottigere",
        "Thanisandra", "Hennur", "Ramamurthy Nagar"
    ]

    all_localities = premium_localities + mid_tier_localities + budget_localities

    # Generate random samples
    data = {
        "location": rng.choice(all_localities, size=n_samples),
        "size": rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5], size=n_samples),  # BHK weighted towards 2-3
        "bath": None,  # Will be derived from BHK
        "balcony": None,  # Will be derived
    }

    df = pd.DataFrame(data)

    # Derive bathrooms (typically = BHK or BHK-1 for larger units)
    df["bath"] = df["size"].apply(
        lambda bhk: max(1, bhk if bhk <= 3 else bhk - rng.integers(0, 2))
    )

    # Derive balconies (0 to min(2, BHK-1))
    df["balcony"] = df["size"].apply(
        lambda bhk: rng.integers(0, min(3, bhk))
    )

    # Generate total_sqft based on BHK
    # Typical: 1 BHK: 500-800, 2 BHK: 900-1300, 3 BHK: 1200-1800, 4 BHK: 1800-2500, 5 BHK: 2500-4000
    sqft_ranges = {
        1: (500, 800),
        2: (900, 1300),
        3: (1200, 1800),
        4: (1800, 2500),
        5: (2500, 4000),
    }

    def sample_sqft(bhk):
        min_sqft, max_sqft = sqft_ranges.get(bhk, (1000, 2000))
        return rng.integers(min_sqft, max_sqft)

    df["total_sqft"] = df["size"].apply(sample_sqft).astype(float)

    # Generate price based on location tier, BHK, and sqft
    # Base price per sqft by locality tier
    def get_base_price_per_sqft(location):
        if location in premium_localities:
            return rng.uniform(6000, 12000)  # ₹6k-12k per sqft
        elif location in mid_tier_localities:
            return rng.uniform(3500, 7000)   # ₹3.5k-7k per sqft
        else:  # budget
            return rng.uniform(2000, 4500)   # ₹2k-4.5k per sqft

    df["price_per_sqft"] = df["location"].apply(get_base_price_per_sqft)

    # Calculate base price
    df["price"] = (df["price_per_sqft"] * df["total_sqft"]).round(0)

    # Add BHK premium (larger BHK slightly higher per sqft)
    df["price"] *= (1 + (df["size"] - 2) * 0.05).clip(lower=0.9, upper=1.3)

    # Add random noise (±15%)
    noise = rng.uniform(0.85, 1.15, size=n_samples)
    df["price"] *= noise

    # Round prices to nearest ₹50,000
    df["price"] = (df["price"] / 50000).round() * 50000

    # Ensure price bounds: ₹10 Lakh to ₹10 Crore
    df["price"] = df["price"].clip(lower=10_00_000, upper=10_00_00_000)

    # Map location to coordinates
    df["latitude"] = df["location"].apply(lambda loc: get_locality_coordinates(loc)[0])
    df["longitude"] = df["location"].apply(lambda loc: get_locality_coordinates(loc)[1])

    # Reorder columns
    df = df[[
        "location", "size", "total_sqft", "bath", "balcony",
        "price", "latitude", "longitude"
    ]]

    logger.info(f"Generated synthetic Bengaluru dataset: {len(df)} properties")
    logger.info(f"Price range: ₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f}")
    logger.info(f"BHK distribution:\n{df['size'].value_counts().sort_index()}")
    logger.info(f"Localities: {df['location'].nunique()} unique areas")

    return df


def load_bengaluru_housing(force_reload: bool = False) -> pd.DataFrame:
    """
    Load Bengaluru Housing dataset.

    If a CSV exists in data/raw/, loads it.
    Otherwise, generates a synthetic dataset and saves it.

    Returns:
        DataFrame with columns: location, size, total_sqft, bath, balcony,
        price, latitude, longitude.
    """
    raw_path = AppConfig.paths.data_raw / "bengaluru_housing.csv"

    # Use cached version if available
    if raw_path.exists() and not force_reload:
        logger.info("Loading cached Bengaluru housing data")
        df = pd.read_csv(raw_path)
        logger.info(f"Loaded {len(df)} properties from {raw_path}")
        return df

    logger.info("Generating synthetic Bengaluru housing dataset")
    df = generate_synthetic_bengaluru_data(n_samples=15000, seed=AppConfig.data.random_seed)

    # Save to CSV
    save_dataframe(df, raw_path)

    logger.info(f"Dataset saved to {raw_path}")
    logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    df = load_bengaluru_housing(force_reload=True)
    print(f"\n{'='*70}")
    print("Bengaluru Housing Dataset Summary")
    print(f"{'='*70}")
    print(f"Shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nPrice statistics (₹):")
    print(df["price"].describe())
    print(f"\nBHK distribution:")
    print(df["size"].value_counts().sort_index())
    print(f"\nTop 10 localities:")
    print(df["location"].value_counts().head(10))
