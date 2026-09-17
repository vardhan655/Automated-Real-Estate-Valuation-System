"""
Schema validation for the loaded Bengaluru dataset.

This module checks for:
- Expected columns present (location, size, total_sqft, bath, balcony, price, latitude, longitude)
- Data types correct
- No unexpected nulls
- Value ranges within plausible bounds for Bengaluru real estate
- Duplicate detection
- Target variable sanity checks (INR price)

Design decision: validation runs early and fails fast, before any
feature engineering or model training contaminates the pipeline.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.utils.logger import get_logger
from src.utils.currency import format_inr

logger = get_logger(__name__)


class ValidationReport:
    """Collects and displays validation results."""

    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        status = "✓" if passed else "✗"
        self.checks.append((name, passed, detail))
        level = logger.info if passed else logger.warning
        level(f"  {status} {name}: {detail}")

    @property
    def all_passed(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def summary(self) -> str:
        passed = sum(1 for _, p, _ in self.checks if p)
        total = len(self.checks)
        return f"{passed}/{total} validation checks passed"


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """
    Run all validation checks on the raw Bengaluru dataset.

    Args:
        df: Raw DataFrame to validate.

    Returns:
        ValidationReport with all check results.
    """
    report = ValidationReport()
    config = AppConfig.data

    logger.info("Running Bengaluru dataset validation checks...")

    # ── 1. Expected columns ──────────────────────────────────────────────
    expected_cols = (
        config.numeric_features
        + config.location_features
        + [config.target_column]
    )
    missing_cols = set(expected_cols) - set(df.columns)
    report.add(
        "Expected columns present",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}" if missing_cols else "All columns found",
    )

    # ── 2. No completely empty columns ───────────────────────────────────
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    report.add(
        "No empty columns",
        len(empty_cols) == 0,
        f"Empty: {empty_cols}" if empty_cols else "None",
    )

    # ── 3. Null analysis ─────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        null_detail = ", ".join(f"{c}: {n}" for c, n in null_cols.items())
        report.add("Null check", True, f"Nulls found (not fatal): {null_detail}")
    else:
        report.add("Null check", True, "No nulls in any column")

    # ── 4. Duplicate rows ────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    report.add(
        "Duplicate rows",
        True,  # duplicates are noted, not blocking
        f"{n_dupes} duplicate rows found" if n_dupes else "No duplicates",
    )

    # ── 5. Target variable checks ────────────────────────────────────────
    target = df[config.target_column]
    report.add(
        "Target is numeric",
        pd.api.types.is_numeric_dtype(target),
        f"dtype={target.dtype}",
    )
    report.add(
        "Target has no negatives",
        (target >= 0).all(),
        f"min={format_inr(target.min())}",
    )
    report.add(
        "Target not constant",
        target.nunique() > 1,
        f"unique values: {target.nunique()}",
    )
    report.add(
        "Price range plausible (₹5L - ₹50Cr)",
        (target >= 500000).all() and (target <= 500000000).all(),
        f"range [{format_inr(target.min())}, {format_inr(target.max())}]",
    )

    # ── 6. Numeric feature ranges ────────────────────────────────────────
    for col in config.numeric_features:
        if col in df.columns:
            has_inf = np.isinf(df[col]).any()
            report.add(
                f"{col}: no infinities",
                not has_inf,
                f"range [{df[col].min():.2f}, {df[col].max():.2f}]",
            )

    # ── 7. Location range check (Bengaluru Metropolitan Region) ──────────
    if "latitude" in df.columns and "longitude" in df.columns:
        lat_ok = df["latitude"].between(12.6, 13.4).all()
        lon_ok = df["longitude"].between(77.2, 78.0).all()
        report.add(
            "Location bounds (Bengaluru Region)",
            lat_ok and lon_ok,
            f"lat: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}], "
            f"lon: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]",
        )

    # ── 8. Dataset size ──────────────────────────────────────────────────
    report.add(
        "Dataset size",
        len(df) >= 1000,
        f"{len(df):,} rows",
    )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info(f"Validation complete: {report.summary()}")
    return report


def get_column_profiles(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Generate detailed profiles for each column.

    Useful for EDA and understanding data quality before pipeline design.
    """
    profiles = {}
    for col in df.columns:
        profile = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean() * 100, 2),
            "unique": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            profile.update({
                "mean": round(df[col].mean(), 4),
                "std": round(df[col].std(), 4),
                "min": round(df[col].min(), 4),
                "max": round(df[col].max(), 4),
                "median": round(df[col].median(), 4),
                "skew": round(df[col].skew(), 4),
            })
        profiles[col] = profile
    return profiles


if __name__ == "__main__":
    from src.data.loader import load_dataset

    df = load_dataset()
    report = validate_dataset(df)
    print(f"\n{report.summary()}")

    profiles = get_column_profiles(df)
    print("\nColumn Profiles:")
    for col, profile in profiles.items():
        print(f"  {col}: {profile}")
