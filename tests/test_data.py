"""Unit tests for data loading, validation, and generation."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.data.validator import validate_dataset, get_column_profiles
from src.data.generator import generate_descriptions, generate_single_description


def test_load_bengaluru_housing():
    """Test that Bengaluru data loads with expected columns and reasonable rows."""
    df = load_dataset()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 10000
    assert "price" in df.columns
    assert "longitude" in df.columns
    assert "latitude" in df.columns
    assert "size" in df.columns
    assert "total_sqft" in df.columns
    assert "location" in df.columns
    assert (df["price"] > 0).all()


def test_validate_dataset():
    """Test that validation runs and catches issues."""
    df = load_dataset()
    report = validate_dataset(df)

    assert report.all_passed
    assert len(report.checks) > 5


def test_generate_descriptions():
    """Test description generation for Bengaluru properties."""
    df = load_dataset().head(10)
    df_with_desc = generate_descriptions(df, seed=42)

    assert "property_description" in df_with_desc.columns
    assert len(df_with_desc) == 10
    assert (df_with_desc["property_description"].str.len() > 20).all()

    # Check reproducibility
    df_with_desc2 = generate_descriptions(df, seed=42)
    assert (df_with_desc["property_description"] == df_with_desc2["property_description"]).all()


def test_column_profiles():
    """Test column profiling."""
    df = load_dataset().head(100)
    profiles = get_column_profiles(df)

    assert "price" in profiles
    assert profiles["price"]["null_count"] == 0
    assert "mean" in profiles["price"]
    assert "location" in profiles
