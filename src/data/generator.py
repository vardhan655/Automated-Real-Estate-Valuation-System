"""
Property description generator.

Generates realistic property descriptions from structured features.

Design decisions:
- Rule-based templates, not LLM-generated, so the process is deterministic
  and reproducible (same seed → same descriptions every time).
- Descriptions encode real feature values in natural language so that
  transformer embeddings can actually learn something useful from them.
- This makes the NLP fusion experiment meaningful: the text genuinely
  contains price-relevant signal, just expressed in language.
- In interviews, explain this as a "feature-aware text augmentation" strategy
  used when real listing descriptions aren't available.

Limitations (state these in interviews):
- Descriptions are synthetic; real listing descriptions would have richer
  vocabulary, marketing language, and neighborhood-specific details.
- The NLP signal partially overlaps with structured features by construction.
  Hybrid improvement shows the model can extract that signal from text.
"""

import random
from typing import List

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Template components ──────────────────────────────────────────────────────

PROPERTY_TYPES = [
    "single-family home", "residential property", "family residence",
    "house", "dwelling", "home",
]

CONDITION_TEMPLATES = {
    "new": ["recently built", "modern construction", "newly constructed"],
    "mid": ["well-maintained", "in good condition", "established"],
    "old": ["classic", "vintage", "character-filled older"],
}

AREA_DESCRIPTIONS = {
    "urban_dense": [
        "in a bustling urban area", "in a densely populated neighborhood",
        "located in a vibrant city block", "in a lively metropolitan area",
    ],
    "urban": [
        "in a well-connected urban location", "in a city neighborhood",
        "situated in a residential urban district",
    ],
    "suburban": [
        "in a quiet suburban neighborhood", "in a family-friendly suburban area",
        "nestled in a peaceful residential community",
    ],
    "rural": [
        "in a spacious rural setting", "in a low-density area with open space",
        "in a tranquil countryside location",
    ],
}

INCOME_DESCRIPTIONS = {
    "high": [
        "affluent neighborhood", "upscale community", "premium location",
        "sought-after area", "high-value district",
    ],
    "mid_high": [
        "desirable area", "growing community", "popular residential zone",
    ],
    "mid": [
        "established community", "stable neighborhood", "accessible area",
    ],
    "low": [
        "affordable area", "developing neighborhood", "budget-friendly location",
    ],
}

ROOM_DESCRIPTIONS = [
    "featuring {rooms:.1f} rooms on average",
    "with approximately {rooms:.1f} rooms per unit",
    "offering {rooms:.1f} rooms per residence",
]

BEDROOM_PHRASES = [
    "including {bedrooms:.1f} bedrooms",
    "with {bedrooms:.1f} bedrooms",
    "{bedrooms:.1f} bedrooms included",
]

COASTAL_PHRASES = [
    "with easy access to the coast", "near the California coastline",
    "within reach of Pacific beaches",
]

INLAND_PHRASES = [
    "in the California interior", "in the inland region",
    "away from the coastline",
]

CLOSING_PHRASES = [
    "A solid choice for buyers looking for value.",
    "Ideal for families and professionals.",
    "A great opportunity in this market.",
    "Well-positioned for long-term value.",
    "Worth considering for its location and features.",
    "A property with real potential.",
]


def _classify_density(population_per_household: float) -> str:
    """Classify area type based on average occupancy."""
    if population_per_household > 5:
        return "urban_dense"
    elif population_per_household > 3:
        return "urban"
    elif population_per_household > 2:
        return "suburban"
    else:
        return "rural"


def _classify_income(median_income: float) -> str:
    """Classify income bracket."""
    if median_income > 6:
        return "high"
    elif median_income > 4:
        return "mid_high"
    elif median_income > 2.5:
        return "mid"
    else:
        return "low"


def _classify_age(age: float) -> str:
    """Classify property condition from housing age."""
    if age < 10:
        return "new"
    elif age < 30:
        return "mid"
    else:
        return "old"


def _is_coastal(longitude: float, latitude: float) -> bool:
    """Rough heuristic: coastal if longitude < -121 and latitude > 34."""
    # Very approximate — California coast runs north-south roughly at -121 to -118
    return longitude < -120.5 or (longitude < -117.5 and latitude < 34)


def generate_single_description(row: pd.Series, rng: random.Random) -> str:
    """
    Generate a natural-language property description from a single row.

    Args:
        row: A row from the housing DataFrame.
        rng: Seeded random.Random instance for reproducibility.

    Returns:
        A 2-4 sentence property description string.
    """
    # Classify features into human-readable categories
    age_class = _classify_age(row["housing_median_age"])
    pop_per_hh = row.get("population_per_household", row["population"] / max(row["households"], 1)) \
        if "population_per_household" not in row.index else row.get("population", 3) / max(row.get("households", 1), 1)
    density_class = _classify_density(pop_per_hh)
    income_class = _classify_income(row["median_income"])
    coastal = _is_coastal(row["longitude"], row["latitude"])

    # Build the description sentence by sentence
    sentences = []

    # Sentence 1: Property type + condition + location
    prop_type = rng.choice(PROPERTY_TYPES)
    condition = rng.choice(CONDITION_TEMPLATES[age_class])
    area_desc = rng.choice(AREA_DESCRIPTIONS[density_class])
    sentences.append(f"This {condition} {prop_type} is located {area_desc}.")

    # Sentence 2: Room and bedroom info
    room_desc = rng.choice(ROOM_DESCRIPTIONS).format(rooms=row["total_rooms"])
    bed_desc = rng.choice(BEDROOM_PHRASES).format(bedrooms=row["total_bedrooms"])
    sentences.append(f"The property {room_desc}, {bed_desc}.")

    # Sentence 3: Neighborhood character (income + coastal)
    neighborhood = rng.choice(INCOME_DESCRIPTIONS[income_class])
    if coastal:
        coast_phrase = rng.choice(COASTAL_PHRASES)
        sentences.append(f"Situated in a {neighborhood}, {coast_phrase}.")
    else:
        inland_phrase = rng.choice(INLAND_PHRASES)
        sentences.append(f"Located in a {neighborhood}, {inland_phrase}.")

    # Sentence 4: Closing (50% chance to keep descriptions varied in length)
    if rng.random() > 0.5:
        sentences.append(rng.choice(CLOSING_PHRASES))

    return " ".join(sentences)


def generate_descriptions(
    df: pd.DataFrame,
    seed: int = None,
) -> pd.Series:
    """
    Generate property descriptions for all rows in the DataFrame.

    Args:
        df: DataFrame with structured housing features.
        seed: Random seed for reproducibility. Defaults to config seed.

    Returns:
        pd.Series of description strings, same index as df.
    """
    if seed is None:
        seed = AppConfig.data.random_seed

    rng = random.Random(seed)
    logger.info(f"Generating property descriptions for {len(df)} listings (seed={seed})")

    descriptions = []
    for idx, row in df.iterrows():
        desc = generate_single_description(row, rng)
        descriptions.append(desc)

    result = pd.Series(descriptions, index=df.index, name="property_description")

    # Log statistics
    lengths = result.str.len()
    word_counts = result.str.split().str.len()
    logger.info(
        f"Description stats — "
        f"chars: {lengths.mean():.0f} avg ({lengths.min()}-{lengths.max()}), "
        f"words: {word_counts.mean():.0f} avg ({word_counts.min()}-{word_counts.max()})"
    )

    return result


if __name__ == "__main__":
    from src.data.loader import load_california_housing

    df = load_california_housing()
    descriptions = generate_descriptions(df)

    print(f"\n{'='*60}")
    print("Sample Generated Descriptions")
    print(f"{'='*60}")
    for i in range(5):
        idx = i * 4000  # sample diverse rows
        print(f"\n--- Row {idx} (price: ${df.iloc[idx]['median_house_value']:,.0f}) ---")
        print(descriptions.iloc[idx])
