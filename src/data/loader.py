"""
Data loader for the House Price Intelligence System (Bengaluru, India).

Uses Bengaluru housing dataset with locality, BHK, sqft, bathrooms, balconies,
coordinates, and INR prices.
"""

import pandas as pd
from src.data.bengaluru_loader import load_bengaluru_housing
from src.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(force_reload: bool = False) -> pd.DataFrame:
    """Load standard Bengaluru housing dataset."""
    return load_bengaluru_housing(force_reload=force_reload)


# Alias for backward compatibility
def load_california_housing(force_reload: bool = False) -> pd.DataFrame:
    """Alias for loading the primary housing dataset."""
    return load_bengaluru_housing(force_reload=force_reload)


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset loaded: {df.shape}")
    print(df.head())
