"""
Centralized configuration for the House Price Intelligence System.

Design decisions:
- Dataclass-based config for type safety and IDE autocomplete.
- All paths are relative to PROJECT_ROOT, making the project portable.
- Random seed set in one place, used everywhere.
- Model hyperparameters stored here to keep training scripts clean.
- Config is importable, not file-based YAML, to keep complexity low for a
  student project while still being structured and centralized.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Project root (two levels up from this file: src/config/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class PathConfig:
    """All project paths, resolved relative to PROJECT_ROOT."""

    root: Path = PROJECT_ROOT

    # Data directories
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"

    # Artifact directories
    models: Path = PROJECT_ROOT / "artifacts" / "models"
    embeddings: Path = PROJECT_ROOT / "artifacts" / "embeddings"
    preprocessors: Path = PROJECT_ROOT / "artifacts" / "preprocessors"
    retrieval: Path = PROJECT_ROOT / "artifacts" / "retrieval"
    reports: Path = PROJECT_ROOT / "artifacts" / "reports"

    # Notebook directory
    notebooks: Path = PROJECT_ROOT / "notebooks"

    def create_all(self) -> None:
        """Create all directories if they don't exist."""
        for field_name in self.__dataclass_fields__:
            path = getattr(self, field_name)
            if isinstance(path, Path) and field_name != "root":
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Dataset and splitting parameters."""

    target_column: str = "price"
    test_size: float = 0.15
    val_size: float = 0.15  # proportion of the remaining train set
    random_seed: int = 42

    # Columns from Bengaluru Housing Dataset
    numeric_features: List[str] = field(
        default_factory=lambda: [
            "size",           # BHK count (1-5)
            "total_sqft",     # Square feet area
            "bath",           # Number of bathrooms
            "balcony",        # Number of balconies
        ]
    )
    location_features: List[str] = field(
        default_factory=lambda: ["longitude", "latitude"]
    )
    locality_column: str = "location"  # Locality name (categorical)
    text_column: str = "property_description"

    # Engineered feature names (created during feature engineering)
    engineered_features: List[str] = field(
        default_factory=lambda: [
            "price_per_sqft",
            "distance_to_cbd",
            "distance_to_tech_hub",
            "distance_to_airport",
            "location_cluster",
        ]
    )

    # Location clustering
    n_location_clusters: int = 30

    # Outlier thresholds (IQR multiplier)
    outlier_iqr_multiplier: float = 3.0


@dataclass
class NLPConfig:
    """NLP and transformer embedding parameters."""

    # DistilBERT: 6x faster than BERT, 97% performance, practical for a student project.
    # RoBERTa is stronger but slower; DistilBERT is the right tradeoff here.
    model_name: str = "distilbert-base-uncased"
    embedding_dim: int = 768
    max_length: int = 128  # property descriptions are short
    batch_size: int = 32
    device: str = "cpu"  # safe default; auto-detected at runtime

    # Embedding reduction via PCA to avoid curse of dimensionality
    # when fusing with ~15 structured features
    pca_components: int = 50


@dataclass
class ModelConfig:
    """Model training hyperparameters."""

    random_seed: int = 42

    # ElasticNet
    elasticnet_alpha: float = 0.1
    elasticnet_l1_ratio: float = 0.5

    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = 20
    rf_min_samples_leaf: int = 5

    # Ridge (baseline)
    ridge_alpha: float = 1.0

    # Optional MLP
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    mlp_dropout: float = 0.2
    mlp_lr: float = 1e-3
    mlp_epochs: int = 100
    mlp_batch_size: int = 64
    mlp_patience: int = 10  # early stopping patience

    # Quantile regression for uncertainty
    quantile_lower: float = 0.1
    quantile_upper: float = 0.9


@dataclass
class RetrievalConfig:
    """Comparable listings retrieval settings."""

    top_k: int = 5
    similarity_metric: str = "cosine"  # cosine works well for normalized embeddings
    feature_space: str = "fused"  # "text", "structured", or "fused"


@dataclass
class ExplainabilityConfig:
    """SHAP and explanation settings."""

    shap_max_samples: int = 200  # limit background dataset for speed
    top_features: int = 10  # number of features to show in explanations


@dataclass
class Config:
    """Master configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)

    def __post_init__(self):
        """Ensure all directories exist on config creation."""
        self.paths.create_all()


# ── Convenience singleton ────────────────────────────────────────────────────
# Import and use: from src.config import AppConfig
AppConfig = Config()
