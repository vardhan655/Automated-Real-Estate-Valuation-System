"""Unit tests for data loading, validation, and generation."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_california_housing
from src.data.validator import validate_dataset, get_column_profiles
from src.data.generator import generate_descriptions, generate_single_description


def test_load_california_housing():
    """Test that data loads with expected columns and reasonable rows."""
    df = load_california_housing()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 10000
    assert "median_house_value" in df.columns
    assert "longitude" in df.columns
    assert "latitude" in df.columns
    assert "housing_median_age" in df.columns
    assert (df["median_house_value"] > 0).all()


def test_validate_dataset():
    """Test that validation runs and catches issues."""
    df = load_california_housing()
    report = validate_dataset(df)

    assert report.all_passed
    assert len(report.checks) > 5


def test_generate_descriptions():
    """Test description generation."""
    df = load_california_housing().head(10)
    descriptions = generate_descriptions(df, seed=42)

    assert len(descriptions) == 10
    assert descriptions.name == "property_description"
    assert (descriptions.str.len() > 20).all()
    # Check reproducibility
    descriptions2 = generate_descriptions(df, seed=42)
    assert (descriptions == descriptions2).all()


def test_column_profiles():
    """Test column profiling."""
    df = load_california_housing().head(100)
    profiles = get_column_profiles(df)

    assert "median_house_value" in profiles
    assert profiles["median_house_value"]["null_count"] == 0
    assert "mean" in profiles["median_house_value"]
