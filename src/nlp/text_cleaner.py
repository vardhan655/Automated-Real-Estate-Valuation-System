"""
Text cleaning for property descriptions.

Design decision: Minimal cleaning because transformers (DistilBERT) have
their own tokenizer that handles casing, punctuation, and subword tokenization.
Heavy cleaning (stripping punctuation, stemming) is counterproductive
for transformer models — it was designed for bag-of-words / TF-IDF.

We only handle:
- Whitespace normalization
- Empty/null value handling
- Extreme-length truncation (the tokenizer handles the rest)
"""

import re
from typing import List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    Minimal text cleaning for transformer input.

    Only normalizes whitespace and handles edge cases.
    Do NOT strip punctuation or lowercase — DistilBERT's tokenizer does that.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "No description available."

    # Normalize whitespace (multiple spaces, tabs, newlines → single space)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_descriptions(descriptions: pd.Series) -> pd.Series:
    """
    Clean a Series of property descriptions.

    Args:
        descriptions: Series of raw description strings.

    Returns:
        Series of cleaned descriptions.
    """
    cleaned = descriptions.apply(clean_text)

    n_empty = (descriptions.isna() | (descriptions.str.strip() == "")).sum()
    if n_empty > 0:
        logger.warning(f"Replaced {n_empty} empty/null descriptions with placeholder")

    logger.info(f"Cleaned {len(cleaned)} descriptions")
    return cleaned
