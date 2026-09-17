"""
End-to-end inference pipeline.

This is the production-style inference interface that:
1. Takes raw input (structured features + description)
2. Runs through all preprocessing/embedding steps
3. Returns predicted price, interval, similar listings, and explanation

Design for reusability: the predictor loads all artifacts once at init,
then predict() can be called repeatedly without reloading.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.features.engineer import engineer_features
from src.features.preprocessor import StructuredPreprocessor, inverse_transform_target
from src.nlp.text_cleaner import clean_text
from src.nlp.embedder import TextEmbedder
from src.models.hybrid import fuse_features
from src.retrieval.similarity import ListingRetriever
from src.explainability.explainer import ModelExplainer
from src.utils.logger import get_logger
from src.utils.io import load_pickle, load_json

logger = get_logger(__name__)


class HousePricePredictor:
    """
    Production inference pipeline for house price prediction.

    Usage:
        predictor = HousePricePredictor()
        result = predictor.predict(property_dict)
        print(result['predicted_price'])
        print(result['interval'])
        print(result['similar_listings'])
    """

    def __init__(
        self,
        model_name: str = "hybrid_random_forest",
        artifacts_dir: Path = None,
    ):
        """
        Load all trained artifacts.

        Args:
            model_name: Name of the model to use for predictions.
            artifacts_dir: Root directory containing artifacts/ folder.
        """
        self.model_name = model_name
        self.artifacts_dir = artifacts_dir or AppConfig.paths.root

        logger.info(f"Loading inference artifacts for model: {model_name}")

        # Load preprocessors
        self.kmeans = load_pickle(AppConfig.paths.preprocessors / "location_kmeans.pkl")
        self.structured_preprocessor = StructuredPreprocessor.load()

        # Load NLP components
        self.embedder = TextEmbedder()
        self.embedder.load_pca()

        # Load model
        self.model = load_pickle(AppConfig.paths.models / f"{model_name}.pkl")

        # Load retrieval index
        self.retriever = ListingRetriever.load()

        # Load explainer (optional)
        try:
            self.explainer = load_pickle(AppConfig.paths.models / "shap_explainer.pkl")
        except FileNotFoundError:
            logger.warning("SHAP explainer not found. Local explanations will be unavailable.")
            self.explainer = None

        # Load interval bounds
        try:
            interval_data = load_json(AppConfig.paths.preprocessors / "interval_bounds.json")
            self.interval_lower = interval_data["lower"]
            self.interval_upper = interval_data["upper"]
        except FileNotFoundError:
            logger.warning("Interval bounds not found. Using default ±$50k interval.")
            self.interval_lower = -50000
            self.interval_upper = 50000

        logger.info("Inference pipeline ready")

    def predict(
        self,
        property_data: Dict,
        return_similar: bool = True,
        return_explanation: bool = True,
        top_k_similar: int = 5,
    ) -> Dict:
        """
        Make a prediction for a single property.

        Args:
            property_data: Dict with keys matching required features:
                - longitude, latitude
                - housing_median_age
                - total_rooms, total_bedrooms
                - population, households
                - median_income
                - property_description (str)
            return_similar: Whether to include similar listings.
            return_explanation: Whether to include SHAP explanation.
            top_k_similar: Number of similar listings to return.

        Returns:
            Dict with:
                - predicted_price: float
                - price_interval: (lower, upper)
                - similar_listings: list of dicts (if return_similar=True)
                - explanation: dict of feature → SHAP value (if return_explanation=True)
        """
        # ── 1. Convert to DataFrame ──────────────────────────────────────
        input_df = pd.DataFrame([property_data])

        # ── 2. Feature engineering ───────────────────────────────────────
        input_eng, _ = engineer_features(
            input_df,
            kmeans_model=self.kmeans,
            is_train=False,
        )

        # ── 3. Structured preprocessing ──────────────────────────────────
        X_struct = self.structured_preprocessor.transform(input_eng)

        # ── 4. Text embedding ────────────────────────────────────────────
        description = clean_text(property_data.get("property_description", ""))
        emb_raw = self.embedder.embed([description])
        emb_pca = self.embedder.transform_reduce(emb_raw)

        # ── 5. Fuse features ─────────────────────────────────────────────
        X_fused = fuse_features(X_struct, emb_pca)

        # ── 6. Predict ───────────────────────────────────────────────────
        y_pred_log = self.model.predict(X_fused)
        predicted_price = float(inverse_transform_target(y_pred_log)[0])

        # ── 7. Interval ──────────────────────────────────────────────────
        price_lower = predicted_price + self.interval_lower
        price_upper = predicted_price + self.interval_upper

        result = {
            "predicted_price": round(predicted_price, 2),
            "price_interval": (round(price_lower, 2), round(price_upper, 2)),
            "model_used": self.model_name,
        }

        # ── 8. Similar listings ──────────────────────────────────────────
        if return_similar:
            similar = self.retriever.query(
                X_fused[0],
                top_k=top_k_similar,
                exclude_self=True,
            )
            result["similar_listings"] = similar

        # ── 9. Explanation ───────────────────────────────────────────────
        if return_explanation and self.explainer is not None:
            explanation = self.explainer.explain_local(X_fused[0])
            # Return only top features for brevity
            top_n = 8
            top_features = dict(list(explanation.items())[:top_n])
            result["explanation"] = top_features

        return result

    def predict_batch(
        self,
        properties: List[Dict],
        return_similar: bool = False,
        return_explanation: bool = False,
    ) -> List[Dict]:
        """
        Predict for multiple properties (batched for efficiency).

        Args:
            properties: List of property dicts.
            return_similar: Whether to include similar listings.
            return_explanation: Whether to include explanations.

        Returns:
            List of result dicts.
        """
        results = []
        for prop in properties:
            result = self.predict(
                prop,
                return_similar=return_similar,
                return_explanation=return_explanation,
            )
            results.append(result)
        return results


if __name__ == "__main__":
    # Demo inference on a sample property
    predictor = HousePricePredictor()

    sample_property = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 21.0,
        "total_rooms": 7.0,
        "total_bedrooms": 3.5,
        "population": 2000,
        "households": 500,
        "median_income": 5.5,
        "property_description": (
            "This well-maintained single-family home is located in a quiet suburban "
            "neighborhood. The property features 7.0 rooms on average, including 3.5 bedrooms. "
            "Situated in a desirable area, with easy access to the coast."
        ),
    }

    result = predictor.predict(sample_property)

    print(f"\n{'='*60}")
    print("INFERENCE DEMO")
    print(f"{'='*60}")
    print(f"Predicted Price: ${result['predicted_price']:,.0f}")
    print(f"Price Interval (80%): ${result['price_interval'][0]:,.0f} – ${result['price_interval'][1]:,.0f}")
    print(f"\nTop Similar Listings:")
    for listing in result.get("similar_listings", [])[:3]:
        price = listing["listing"].get("median_house_value", "N/A")
        similarity = listing["similarity"]
        print(f"  #{listing['rank']}: ${price:,.0f} (similarity: {similarity:.2%})")
    print(f"\nTop Explanation Features:")
    for feat, val in list(result.get("explanation", {}).items())[:5]:
        direction = "↑" if val > 0 else "↓"
        print(f"  {feat}: {val:+.4f} {direction}")
    print(f"{'='*60}")
