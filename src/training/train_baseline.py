"""
Baseline model training pipeline.

Runs the full structured-only training workflow:
1. Load and validate data
2. Split into train/val/test
3. Engineer features
4. Preprocess
5. Train all baseline models
6. Evaluate and compare
7. Save artifacts

This is the script you run first to establish the baseline before
adding NLP features.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import AppConfig
from src.data.loader import load_california_housing
from src.data.validator import validate_dataset
from src.data.generator import generate_descriptions
from src.features.engineer import engineer_features
from src.features.preprocessor import (
    split_data,
    StructuredPreprocessor,
    prepare_target,
    inverse_transform_target,
)
from src.models.baseline import train_all_baselines
from src.training.evaluator import compute_metrics, compare_models, plot_predictions_vs_actual, plot_residuals
from src.utils.logger import get_logger
from src.utils.io import save_pickle, save_dataframe, save_metrics

logger = get_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("PHASE 1: Baseline Model Training Pipeline")
    logger.info("=" * 60)

    # ── 1. Load and validate ─────────────────────────────────────────────
    logger.info("\n── Step 1: Data Loading & Validation ──")
    df = load_california_housing()
    report = validate_dataset(df)

    if not report.all_passed:
        logger.warning("Some validation checks failed — review before continuing")

    # ── 2. Generate descriptions (save for later hybrid training) ────────
    logger.info("\n── Step 2: Generate Descriptions ──")
    descriptions = generate_descriptions(df)
    df[AppConfig.data.text_column] = descriptions
    save_dataframe(df, AppConfig.paths.data_interim / "housing_with_descriptions.csv")

    # ── 3. Split data ────────────────────────────────────────────────────
    logger.info("\n── Step 3: Train/Val/Test Split ──")
    train_df, val_df, test_df = split_data(df)

    save_dataframe(train_df, AppConfig.paths.data_processed / "train.csv")
    save_dataframe(val_df, AppConfig.paths.data_processed / "val.csv")
    save_dataframe(test_df, AppConfig.paths.data_processed / "test.csv")

    # ── 4. Feature engineering ───────────────────────────────────────────
    logger.info("\n── Step 4: Feature Engineering ──")
    train_eng, kmeans_model = engineer_features(train_df, is_train=True)
    val_eng, _ = engineer_features(val_df, kmeans_model=kmeans_model, is_train=False)
    test_eng, _ = engineer_features(test_df, kmeans_model=kmeans_model, is_train=False)

    save_pickle(kmeans_model, AppConfig.paths.preprocessors / "location_kmeans.pkl")

    # ── 5. Preprocess structured features ────────────────────────────────
    logger.info("\n── Step 5: Structured Preprocessing ──")
    preprocessor = StructuredPreprocessor()
    X_train = preprocessor.fit_transform(train_eng)
    X_val = preprocessor.transform(val_eng)
    X_test = preprocessor.transform(test_eng)

    preprocessor.save()

    # Prepare targets (log-transformed)
    y_train = prepare_target(train_eng[AppConfig.data.target_column])
    y_val = prepare_target(val_eng[AppConfig.data.target_column])
    y_test = prepare_target(test_eng[AppConfig.data.target_column])

    # Also keep original-scale targets for evaluation
    y_val_orig = val_eng[AppConfig.data.target_column].values
    y_test_orig = test_eng[AppConfig.data.target_column].values

    # ── 6. Train baselines ───────────────────────────────────────────────
    logger.info("\n── Step 6: Train Baseline Models ──")
    fitted_models = train_all_baselines(X_train, y_train)

    # ── 7. Evaluate on validation set ────────────────────────────────────
    logger.info("\n── Step 7: Evaluation ──")
    all_results = {}

    for name, model in fitted_models.items():
        logger.info(f"\nEvaluating {name}:")
        y_pred_log = model.predict(X_val)
        y_pred = inverse_transform_target(y_pred_log)

        metrics = compute_metrics(y_val_orig, y_pred, prefix="val_")
        all_results[name] = metrics

        # Save plots for each model
        plot_predictions_vs_actual(
            y_val_orig, y_pred, name,
            save_path=AppConfig.paths.reports / f"pred_vs_actual_{name}.png"
        )
        plot_residuals(
            y_val_orig, y_pred, name,
            save_path=AppConfig.paths.reports / f"residuals_{name}.png"
        )

    # ── 8. Comparison ────────────────────────────────────────────────────
    logger.info("\n── Step 8: Model Comparison ──")
    compare_models(
        all_results,
        save_path=AppConfig.paths.reports / "baseline_comparison.json"
    )

    # Save metrics
    save_metrics(all_results, AppConfig.paths.reports / "baseline_metrics.json")

    # ── Summary ──────────────────────────────────────────────────────────
    best_model = min(all_results, key=lambda k: all_results[k]["val_rmse"])
    best_rmse = all_results[best_model]["val_rmse"]
    best_r2 = all_results[best_model]["val_r2"]

    logger.info(f"\n{'='*60}")
    logger.info(f"BASELINE TRAINING COMPLETE")
    logger.info(f"Best model: {best_model}")
    logger.info(f"  Val RMSE: ${best_rmse:,.0f}")
    logger.info(f"  Val R²: {best_r2:.4f}")
    logger.info(f"{'='*60}")

    return fitted_models, all_results


if __name__ == "__main__":
    main()
