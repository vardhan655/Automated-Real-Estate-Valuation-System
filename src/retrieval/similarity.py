"""
Comparable listings retrieval system.

Given a property (its features + description), find the top-K most similar
listings from the dataset. This is useful for:
- Market analysis ("what are similar properties priced at?")
- Price validation ("is this prediction in line with comparables?")
- Demo/interview storytelling ("here's why we predicted this price")

Design decisions:
- sklearn NearestNeighbors over FAISS: for ~20K listings, sklearn is
  fast enough and avoids an extra dependency. FAISS would help at 1M+.
- Cosine similarity on fused features: captures both structural similarity
  and semantic text similarity.
- The retrieval index stores original data alongside vectors so we can
  explain WHY listings are similar.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.config import AppConfig
from src.utils.logger import get_logger
from src.utils.io import save_pickle, load_pickle

logger = get_logger(__name__)


class ListingRetriever:
    """
    Retrieves comparable listings using nearest-neighbor search.

    Stores the feature matrix + original listing data so that retrieved
    results come with full context (price, description, location, etc.).
    """

    def __init__(self, config=None):
        self.config = config or AppConfig.retrieval
        self.nn_model: Optional[NearestNeighbors] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.listings_df: Optional[pd.DataFrame] = None
        self.is_fitted = False

    def fit(
        self,
        feature_matrix: np.ndarray,
        listings_df: pd.DataFrame,
    ) -> "ListingRetriever":
        """
        Build the retrieval index.

        Args:
            feature_matrix: (N, D) feature matrix (fused structured + text).
            listings_df: DataFrame with original listing details.

        Returns:
            self for chaining.
        """
        if feature_matrix.shape[0] != len(listings_df):
            raise ValueError(
                f"Feature matrix ({feature_matrix.shape[0]}) and listings "
                f"({len(listings_df)}) must have same number of rows"
            )

        # L2-normalize for cosine similarity via dot product
        self.feature_matrix = normalize(feature_matrix, norm="l2")
        self.listings_df = listings_df.reset_index(drop=True)

        # Build nearest neighbors index
        self.nn_model = NearestNeighbors(
            n_neighbors=self.config.top_k + 1,  # +1 because query might be in the index
            metric="cosine",
            algorithm="brute",  # brute is fine for <100K samples
        )
        self.nn_model.fit(self.feature_matrix)

        self.is_fitted = True
        logger.info(f"Retrieval index built: {feature_matrix.shape[0]} listings, "
                     f"{feature_matrix.shape[1]} features")

        return self

    def query(
        self,
        query_features: np.ndarray,
        top_k: int = None,
        exclude_self: bool = True,
    ) -> List[Dict]:
        """
        Find top-K similar listings for a query.

        Args:
            query_features: (1, D) or (D,) feature vector for the query property.
            top_k: Number of results to return.
            exclude_self: If True, skip exact matches (distance=0).

        Returns:
            List of dicts, each with keys:
                - rank: 1-indexed rank
                - similarity: cosine similarity score (0 to 1)
                - index: original DataFrame index
                - listing: dict of listing details
        """
        if not self.is_fitted:
            raise RuntimeError("Retriever not fitted. Call .fit() first.")

        top_k = top_k or self.config.top_k

        # Reshape to 2D if needed
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)

        # Normalize query
        query_normalized = normalize(query_features, norm="l2")

        # Find neighbors (request extra in case we need to exclude self)
        distances, indices = self.nn_model.kneighbors(query_normalized)

        results = []
        rank = 0
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1.0 - dist  # cosine distance → similarity

            # Skip exact self-match
            if exclude_self and similarity > 0.9999:
                continue

            rank += 1
            if rank > top_k:
                break

            listing_row = self.listings_df.iloc[idx]
            listing_dict = listing_row.to_dict()

            results.append({
                "rank": rank,
                "similarity": round(float(similarity), 4),
                "index": int(idx),
                "listing": listing_dict,
            })

        return results

    def format_results(self, results: List[Dict]) -> str:
        """
        Format retrieval results as a readable string.

        Useful for the Streamlit demo and debugging.
        """
        lines = ["Similar Properties:"]
        lines.append("=" * 70)

        for r in results:
            listing = r["listing"]
            price = listing.get("median_house_value", "N/A")
            desc = listing.get("property_description", "N/A")
            lat = listing.get("latitude", "N/A")
            lon = listing.get("longitude", "N/A")

            lines.append(f"\n  #{r['rank']} — Similarity: {r['similarity']:.2%}")
            lines.append(f"  Price: ${price:,.0f}" if isinstance(price, (int, float)) else f"  Price: {price}")
            lines.append(f"  Location: ({lat}, {lon})")
            lines.append(f"  Income: {listing.get('median_income', 'N/A')}")

            # Truncate description for display
            if isinstance(desc, str) and len(desc) > 120:
                desc = desc[:120] + "..."
            lines.append(f"  Description: {desc}")
            lines.append(f"  {'-'*60}")

        return "\n".join(lines)

    def save(self, path: Path = None) -> None:
        """Save retriever state to disk."""
        path = path or (AppConfig.paths.retrieval / "listing_retriever.pkl")
        save_pickle(self, path)

    @classmethod
    def load(cls, path: Path = None) -> "ListingRetriever":
        """Load retriever from disk."""
        path = path or (AppConfig.paths.retrieval / "listing_retriever.pkl")
        return load_pickle(path)
