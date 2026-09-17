"""Unit tests for models and retrieval."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import build_baseline_models, train_model
from src.models.hybrid import fuse_features, build_hybrid_models
from src.retrieval.similarity import ListingRetriever


def test_baseline_models_build():
    """Test that all baseline models can be instantiated."""
    models = build_baseline_models()
    assert "linear_regression" in models
    assert "ridge" in models
    assert "elasticnet" in models
    assert "random_forest" in models


def test_train_baseline_model():
    """Test training a simple baseline model."""
    X = np.random.randn(50, 10)
    y = np.random.randn(50)

    models = build_baseline_models()
    model = train_model(models["elasticnet"], X, y, "test_elasticnet")

    preds = model.predict(X)
    assert preds.shape == (50,)


def test_feature_fusion():
    """Test early fusion (concatenation)."""
    X_struct = np.random.randn(100, 20)
    X_text = np.random.randn(100, 50)

    X_fused = fuse_features(X_struct, X_text)
    assert X_fused.shape == (100, 70)


def test_listing_retriever():
    """Test nearest-neighbor retrieval."""
    X = np.random.randn(50, 10)
    df = pd.DataFrame({
        "median_house_value": np.random.uniform(100000, 500000, 50),
        "latitude": np.random.uniform(32, 42, 50),
        "longitude": np.random.uniform(-125, -114, 50),
        "property_description": ["Nice house"] * 50,
    })

    retriever = ListingRetriever()
    retriever.fit(X, df)

    query = X[0]
    results = retriever.query(query, top_k=3, exclude_self=False)

    assert len(results) == 3
    assert results[0]["similarity"] > 0.99  # exact match should have high similarity
    assert "listing" in results[0]
    assert "rank" in results[0]
