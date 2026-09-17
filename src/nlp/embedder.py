"""
Transformer-based embedding generator for property descriptions.

Design decisions:
- DistilBERT chosen over RoBERTa: 6x faster, 60% smaller, retains 97% of BERT
  performance. For a student project where embeddings are a feature input (not
  fine-tuned for a specific task), this is the right speed/quality tradeoff.
- CLS token pooling: uses the [CLS] token's hidden state as the sentence
  embedding. Simple, standard, works well for downstream regression.
- Embeddings are 768-dimensional; we optionally reduce to ~50 dims via PCA
  to avoid overwhelming the ~15 structured features in fusion.
- Batch processing for memory efficiency.
- Embeddings are cached to disk so they only need to be computed once.

Interview talking points:
- "We use DistilBERT as a frozen feature extractor, not fine-tuned,
   because our dataset is too small to train a transformer end-to-end
   for regression without severe overfitting."
- "CLS pooling gives a single 768-dim vector per description.
   Mean pooling is an alternative but CLS is standard for classification heads."
- "PCA reduction from 768 → 50 dims prevents the text features from
   dominating the fused representation numerically."
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.config import AppConfig
from src.utils.logger import get_logger
from src.utils.io import save_numpy, load_numpy, save_pickle, load_pickle

logger = get_logger(__name__)


class TextEmbedder:
    """
    Generates fixed-length embeddings from text using a pretrained transformer.

    Usage:
        embedder = TextEmbedder()
        embeddings = embedder.embed(descriptions_list)      # (N, 768)
        reduced = embedder.fit_reduce(embeddings)            # (N, 50)
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or AppConfig.nlp.model_name
        self.max_length = AppConfig.nlp.max_length
        self.batch_size = AppConfig.nlp.batch_size
        self.pca_components = AppConfig.nlp.pca_components

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self.pca: Optional[PCA] = None
        self._is_loaded = False

    def _load_model(self):
        """Lazy-load the transformer model and tokenizer."""
        if self._is_loaded:
            return

        logger.info(f"Loading {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    @torch.no_grad()
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate CLS-token embeddings for a list of texts.

        Args:
            texts: List of description strings.

        Returns:
            numpy array of shape (len(texts), 768).
        """
        self._load_model()

        all_embeddings = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(0, len(texts), self.batch_size),
                      desc="Generating embeddings", total=n_batches):
            batch_texts = texts[i : i + self.batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Forward pass
            outputs = self.model(**encoded)

            # CLS token embedding (first token of last hidden state)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated embeddings: {embeddings.shape}")
        return embeddings

    def fit_reduce(
        self, embeddings: np.ndarray, n_components: int = None
    ) -> np.ndarray:
        """
        Fit PCA on embeddings and return reduced version.

        Use on training embeddings only! Call transform_reduce for val/test.

        Args:
            embeddings: (N, 768) embedding matrix.
            n_components: Number of PCA components (default from config).

        Returns:
            (N, n_components) reduced embedding matrix.
        """
        n_components = n_components or self.pca_components

        self.pca = PCA(n_components=n_components, random_state=AppConfig.data.random_seed)
        reduced = self.pca.fit_transform(embeddings)

        explained_var = self.pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA reduction: {embeddings.shape[1]} → {n_components} dims, "
            f"explained variance: {explained_var:.1%}"
        )

        return reduced

    def transform_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply pre-fitted PCA to new embeddings.

        Args:
            embeddings: (N, 768) embedding matrix.

        Returns:
            (N, n_components) reduced embedding matrix.
        """
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit_reduce on training data first.")
        return self.pca.transform(embeddings)

    def save_pca(self, path: Path = None) -> None:
        """Save fitted PCA model."""
        path = path or (AppConfig.paths.preprocessors / "embedding_pca.pkl")
        save_pickle(self.pca, path)

    def load_pca(self, path: Path = None) -> None:
        """Load fitted PCA model."""
        path = path or (AppConfig.paths.preprocessors / "embedding_pca.pkl")
        self.pca = load_pickle(path)


def generate_and_save_embeddings(
    texts: List[str],
    split_name: str,
    embedder: TextEmbedder = None,
) -> np.ndarray:
    """
    Generate embeddings and save them to disk.

    Args:
        texts: List of description strings.
        split_name: "train", "val", or "test" (used in filename).
        embedder: TextEmbedder instance (creates one if None).

    Returns:
        Raw embedding matrix (N, 768).
    """
    if embedder is None:
        embedder = TextEmbedder()

    embeddings = embedder.embed(texts)

    # Save raw embeddings
    save_path = AppConfig.paths.embeddings / f"embeddings_{split_name}.npy"
    save_numpy(embeddings, save_path)

    return embeddings


if __name__ == "__main__":
    # Quick smoke test with a few descriptions
    sample_texts = [
        "This well-maintained single-family home is located in a quiet suburban neighborhood.",
        "A modern construction residential property in a bustling urban area with 6.5 rooms.",
        "Classic vintage dwelling in an affordable area, in the California interior.",
    ]

    embedder = TextEmbedder()
    embs = embedder.embed(sample_texts)
    print(f"\nEmbedding shape: {embs.shape}")
    print(f"Embedding norms: {np.linalg.norm(embs, axis=1)}")

    reduced = embedder.fit_reduce(embs, n_components=2)
    print(f"Reduced shape: {reduced.shape}")
