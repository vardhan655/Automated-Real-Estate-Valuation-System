"""
Hybrid model training pipeline.

Builds on top of the baseline pipeline by adding NLP embeddings:
1. Load pre-processed data splits
2. Generate transformer embeddings for descriptions
3. PCA-reduce embeddings
4. Fuse with structured features
5. Train hybrid models (ElasticNet + RF on fused features)
6. Optionally train MLP
7. Compare baseline vs hybrid models
8. Build retrieval index
9. Generate SHAP explanations
10. Estimate prediction intervals

This is the main training script that produces the final model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import AppConfig
from src.data.generator import generate_descriptions
from src.features.engineer import engineer_features
from src.features.preprocessor import (
    StructuredPreprocessor,
    prepare_target,
    inverse_transform_target,
)
from src.nlp.text_cleaner import clean_descriptions
from src.nlp.embedder import TextEmbedder, generate_and_save_embeddings
from src.models.hybrid import fuse_features, train_hybrid_models
from src.models.mlp import MLPTrainer
from src.retrieval.similarity import ListingRetriever
from src.explainability.explainer import ModelExplainer
from src.training.evaluator import (
    compute_metrics,
    compare_models,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_model_comparison,
)
from src.utils.logger import get_logger
from src.utils.io import (
    load_dataframe,
    load_pickle,
    save_pickle,
    save_numpy,
    load_numpy,
    save_metrics,
    save_json,
)

logger = get_logger(__name__)


def load_baseline_results() -> dict:
    """Load baseline metrics from previous pipeline run."""
    from src.utils.io import load_json
    try:
        return load_json(AppConfig.paths.reports / "baseline_metrics.json")
    except FileNotFoundError:
        logger.warning("No baseline metrics found. Run train_baseline.py first.")
        return {}


def main():
    logger.info("=" * 60)
    logger.info("PHASE 2: Hybrid Model Training Pipeline")
    logger.info("=" * 60)

    # ── 1. Load processed splits ─────────────────────────────────────────
    logger.info("\n── Step 1: Load Data Splits ──")
    train_df = load_dataframe(AppConfig.paths.data_processed / "train.csv")
    val_df = load_dataframe(AppConfig.paths.data_processed / "val.csv")
    test_df = load_dataframe(AppConfig.paths.data_processed / "test.csv")

    # ── 2. Load structured features ──────────────────────────────────────
    logger.info("\n── Step 2: Rebuild Structured Features ──")
    kmeans_model = load_pickle(AppConfig.paths.preprocessors / "location_kmeans.pkl")
    preprocessor = StructuredPreprocessor.load()

    train_eng, _ = engineer_features(train_df, kmeans_model=kmeans_model, is_train=False)
    val_eng, _ = engineer_features(val_df, kmeans_model=kmeans_model, is_train=False)
    test_eng, _ = engineer_features(test_df, kmeans_model=kmeans_model, is_train=False)

    X_train_struct = preprocessor.transform(train_eng)
    X_val_struct = preprocessor.transform(val_eng)
    X_test_struct = preprocessor.transform(test_eng)

    # Targets
    y_train = prepare_target(train_eng[AppConfig.data.target_column])
    y_val = prepare_target(val_eng[AppConfig.data.target_column])
    y_test = prepare_target(test_eng[AppConfig.data.target_column])
    y_val_orig = val_eng[AppConfig.data.target_column].values
    y_test_orig = test_eng[AppConfig.data.target_column].values

    # ── 3. Generate text embeddings ──────────────────────────────────────
    logger.info("\n── Step 3: NLP Embeddings ──")
    text_col = AppConfig.data.text_column

    embedder = TextEmbedder()

    # Check for cached embeddings
    emb_train_path = AppConfig.paths.embeddings / "embeddings_train.npy"
    emb_val_path = AppConfig.paths.embeddings / "embeddings_val.npy"
    emb_test_path = AppConfig.paths.embeddings / "embeddings_test.npy"

    if emb_train_path.exists() and emb_val_path.exists() and emb_test_path.exists():
        logger.info("Loading cached embeddings")
        emb_train = load_numpy(emb_train_path)
        emb_val = load_numpy(emb_val_path)
        emb_test = load_numpy(emb_test_path)
    else:
        # Clean descriptions
        train_texts = clean_descriptions(train_df[text_col]).tolist()
        val_texts = clean_descriptions(val_df[text_col]).tolist()
        test_texts = clean_descriptions(test_df[text_col]).tolist()

        # Generate embeddings
        emb_train = generate_and_save_embeddings(train_texts, "train", embedder)
        emb_val = generate_and_save_embeddings(val_texts, "val", embedder)
        emb_test = generate_and_save_embeddings(test_texts, "test", embedder)

    # ── 4. PCA reduce embeddings ─────────────────────────────────────────
    logger.info("\n── Step 4: PCA Reduction ──")
    emb_train_pca = embedder.fit_reduce(emb_train)
    emb_val_pca = embedder.transform_reduce(emb_val)
    emb_test_pca = embedder.transform_reduce(emb_test)
    embedder.save_pca()

    save_numpy(emb_train_pca, AppConfig.paths.embeddings / "embeddings_train_pca.npy")
    save_numpy(emb_val_pca, AppConfig.paths.embeddings / "embeddings_val_pca.npy")
    save_numpy(emb_test_pca, AppConfig.paths.embeddings / "embeddings_test_pca.npy")

    # ── 5. Fuse features ─────────────────────────────────────────────────
    logger.info("\n── Step 5: Feature Fusion ──")
    X_train_fused = fuse_features(X_train_struct, emb_train_pca)
    X_val_fused = fuse_features(X_val_struct, emb_val_pca)
    X_test_fused = fuse_features(X_test_struct, emb_test_pca)

    # ── 6. Train hybrid models ───────────────────────────────────────────
    logger.info("\n── Step 6: Train Hybrid Models ──")
    hybrid_models = train_hybrid_models(X_train_fused, y_train)

    # ── 7. Optionally train MLP ──────────────────────────────────────────
    logger.info("\n── Step 7: MLP Experiment ──")
    mlp_trainer = MLPTrainer()
    mlp_trainer.train(X_train_fused, y_train, X_val_fused, y_val)
    mlp_trainer.save()
    hybrid_models["hybrid_mlp"] = mlp_trainer

    # ── 8. Evaluate all models ───────────────────────────────────────────
    logger.info("\n── Step 8: Evaluation ──")

    # Load baseline results for comparison
    baseline_results = load_baseline_results()
    all_results = dict(baseline_results)

    for name, model in hybrid_models.items():
        logger.info(f"\nEvaluating {name}:")
        if isinstance(model, MLPTrainer):
            y_pred_log = model.predict(X_val_fused)
        else:
            y_pred_log = model.predict(X_val_fused)

        y_pred = inverse_transform_target(y_pred_log)
        metrics = compute_metrics(y_val_orig, y_pred, prefix="val_")
        all_results[name] = metrics

        plot_predictions_vs_actual(
            y_val_orig, y_pred, name,
            save_path=AppConfig.paths.reports / f"pred_vs_actual_{name}.png"
        )

    compare_models(all_results, save_path=AppConfig.paths.reports / "full_comparison.json")
    plot_model_comparison(all_results, "val_rmse",
                          save_path=AppConfig.paths.reports / "model_comparison_rmse.png")
    plot_model_comparison(all_results, "val_r2",
                          save_path=AppConfig.paths.reports / "model_comparison_r2.png")

    # ── 9. Build retrieval index ─────────────────────────────────────────
    logger.info("\n── Step 9: Build Retrieval Index ──")
    retriever = ListingRetriever()
    retriever.fit(X_train_fused, train_df)
    retriever.save()

    # Demo retrieval for a sample property
    sample_idx = 0
    results = retriever.query(X_val_fused[sample_idx])
    logger.info(f"\nSample retrieval for validation listing #{sample_idx}:")
    print(retriever.format_results(results))

    # ── 10. Explainability ───────────────────────────────────────────────
    logger.info("\n── Step 10: Explainability ──")

    # Build feature names for fused features
    n_text_features = emb_train_pca.shape[1]
    text_feature_names = [f"text_pca_{i}" for i in range(n_text_features)]
    all_feature_names = preprocessor.feature_names + text_feature_names

    # Use Random Forest for SHAP (TreeExplainer is fast and exact)
    best_hybrid_rf = hybrid_models.get("hybrid_random_forest")
    if best_hybrid_rf is not None:
        explainer = ModelExplainer(
            feature_names=all_feature_names,
            n_text_features=n_text_features,
        )
        explainer.fit(best_hybrid_rf, X_train_fused, model_type="tree")

        # Global importance
        importance = explainer.explain_global(
            X_val_fused[:200],
            save_path=AppConfig.paths.reports / "shap_importance.png"
        )
        save_json(importance, AppConfig.paths.reports / "feature_importance.json")

        # Local explanation for a sample
        local_exp = explainer.explain_local(X_val_fused[0], instance_idx=0)

        # Text contribution summary
        text_summary = explainer.get_text_contribution_summary()
        logger.info(f"\nText vs Structured contribution: {text_summary}")
        save_json(text_summary, AppConfig.paths.reports / "text_contribution.json")

        save_pickle(explainer, AppConfig.paths.models / "shap_explainer.pkl")

    # ── 11. Prediction intervals (residual-based) ────────────────────────
    logger.info("\n── Step 11: Prediction Intervals ──")

    # Use the best model for interval estimation
    best_model_name = min(all_results, key=lambda k: all_results[k].get("val_rmse", float("inf")))
    best_model = hybrid_models.get(best_model_name)

    if best_model is not None:
        if isinstance(best_model, MLPTrainer):
            y_val_pred_log = best_model.predict(X_val_fused)
        else:
            y_val_pred_log = best_model.predict(X_val_fused)

        y_val_pred = inverse_transform_target(y_val_pred_log)
        residuals = y_val_orig - y_val_pred

        # Compute percentile-based intervals from residual distribution
        lower_pct = AppConfig.model.quantile_lower * 100  # 10th percentile
        upper_pct = AppConfig.model.quantile_upper * 100  # 90th percentile
        residual_lower = np.percentile(residuals, lower_pct)
        residual_upper = np.percentile(residuals, upper_pct)

        interval_stats = {
            "method": "residual_percentile",
            "residual_lower_bound": float(residual_lower),
            "residual_upper_bound": float(residual_upper),
            "interval_width_avg": float(residual_upper - residual_lower),
            "nominal_coverage": f"{upper_pct - lower_pct:.0f}%",
            "actual_coverage": float(
                np.mean((residuals >= residual_lower) & (residuals <= residual_upper)) * 100
            ),
            "limitation": "Assumes homoscedastic residuals. Real prediction intervals "
                          "should be wider for high-value properties. A future improvement "
                          "would use quantile regression or conformal prediction.",
        }

        save_json(interval_stats, AppConfig.paths.reports / "interval_stats.json")
        save_json(
            {"lower": float(residual_lower), "upper": float(residual_upper)},
            AppConfig.paths.preprocessors / "interval_bounds.json"
        )

        logger.info(f"  Prediction interval: [{residual_lower:+,.0f}, {residual_upper:+,.0f}]")
        logger.info(f"  Interval width: ${residual_upper - residual_lower:,.0f}")
        logger.info(f"  Actual coverage on val set: {interval_stats['actual_coverage']:.1f}%")

    # ── Final summary ────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("HYBRID TRAINING COMPLETE")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"  Val RMSE: ${all_results[best_model_name]['val_rmse']:,.0f}")
    logger.info(f"  Val R²: {all_results[best_model_name]['val_r2']:.4f}")
    logger.info(f"{'='*60}")

    # Save final metadata
    save_json({
        "best_model": best_model_name,
        "n_structured_features": X_train_struct.shape[1],
        "n_text_features": n_text_features,
        "n_fused_features": X_train_fused.shape[1],
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "embedding_model": AppConfig.nlp.model_name,
        "all_results": all_results,
    }, AppConfig.paths.reports / "training_metadata.json")

    return hybrid_models, all_results


if __name__ == "__main__":
    main()
