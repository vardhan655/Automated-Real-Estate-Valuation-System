"""
Property description generator for Bengaluru real estate.

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

# ── Template components for Bengaluru real estate ────────────────────────────

PROPERTY_TYPES = [
    "apartment", "flat", "independent house", "villa",
    "residential property", "duplex", "builder floor",
]

BHK_DESCRIPTIONS = {
    1: ["compact 1 BHK", "cozy 1 BHK", "well-designed 1 BHK", "modern 1 BHK"],
    2: ["spacious 2 BHK", "comfortable 2 BHK", "well-planned 2 BHK", "modern 2 BHK"],
    3: ["spacious 3 BHK", "premium 3 BHK", "luxurious 3 BHK", "well-appointed 3 BHK"],
    4: ["expansive 4 BHK", "premium 4 BHK", "luxurious 4 BHK", "ultra-spacious 4 BHK"],
    5: ["magnificent 5 BHK", "ultra-luxury 5 BHK", "grand 5 BHK", "palatial 5 BHK"],
}

AREA_SIZE_DESCRIPTIONS = {
    "compact": ["efficient use of space", "smartly designed", "optimized layout"],
    "medium": ["well-proportioned", "comfortable layout", "good space utilization"],
    "spacious": ["generous living space", "sprawling layout", "ample room"],
    "luxury": ["expansive interiors", "palatial space", "ultra-luxurious dimensions"],
}

AMENITIES = [
    "24/7 security", "covered parking", "clubhouse", "swimming pool",
    "gym", "children's play area", "power backup", "lift",
    "landscaped gardens", "jogging track", "indoor games room",
    "community hall", "intercom facility",
]

LOCATION_DESCRIPTIONS = {
    "premium": [
        "located in the heart of {location}", "in the prestigious {location} area",
        "situated in prime {location}", "nestled in sought-after {location}",
    ],
    "good": [
        "located in {location}", "situated in {location}",
        "in the established {location} locality", "in {location}",
    ],
    "emerging": [
        "in the upcoming {location} area", "in the developing {location} locality",
        "strategically located in {location}",
    ],
}

IT_HUB_PROXIMITY = [
    "close to major IT parks", "near tech corridor", "minutes from IT hubs",
    "walking distance to tech offices", "near Whitefield IT zone",
    "close to Electronic City", "near major tech companies",
    "excellent connectivity to IT hubs",
]

TRANSPORT_CONNECTIVITY = [
    "excellent Metro connectivity", "near Metro station",
    "close to Outer Ring Road", "great public transport access",
    "well-connected by road", "easy access to Hosur Road",
    "near major transport routes", "excellent connectivity",
]

BUILDING_QUALITY = {
    "budget": ["decent construction quality", "functional design", "ready to move in"],
    "mid": ["good construction quality", "modern amenities", "quality fittings"],
    "premium": ["premium construction", "high-end fittings", "luxury finishes"],
    "luxury": ["ultra-luxury construction", "imported fittings", "world-class amenities"],
}

BALCONY_PHRASES = [
    "with {balcony} balcony", "featuring {balcony} balcony",
    "{balcony} balcony included", "includes {balcony} balcony",
]

BATHROOM_PHRASES = [
    "with {bath} bathrooms", "{bath} modern bathrooms",
    "featuring {bath} well-appointed bathrooms",
]

CLOSING_PHRASES = [
    "Ideal for families and working professionals.",
    "Perfect for those seeking comfort and convenience.",
    "Great investment opportunity in a prime location.",
    "Excellent choice for modern urban living.",
    "A property that combines comfort with connectivity.",
    "Worth considering for its location and amenities.",
]


def _classify_area_size(total_sqft: float) -> str:
    """Classify property size."""
    if total_sqft < 800:
        return "compact"
    elif total_sqft < 1200:
        return "medium"
    elif total_sqft < 1800:
        return "spacious"
    else:
        return "luxury"


def _classify_price_segment(price: float) -> str:
    """Classify price segment in INR."""
    # Price in rupees
    if price < 4000000:  # < 40 Lakh
        return "budget"
    elif price < 8000000:  # 40L - 80L
        return "mid"
    elif price < 15000000:  # 80L - 1.5 Cr
        return "premium"
    else:  # > 1.5 Cr
        return "luxury"


def _classify_location_tier(location: str) -> str:
    """Classify location into premium/good/emerging tiers."""
    premium_areas = [
        "Koramangala", "Indiranagar", "Jayanagar", "JP Nagar", "HSR Layout",
        "Whitefield", "Marathahalli", "MG Road", "Brigade Road", "Malleshwaram",
        "Basavanagudi", "Rajajinagar", "Sadashiva Nagar", "Yelahanka New Town",
    ]

    emerging_areas = [
        "Sarjapur Road", "Hosa Road", "Jigani", "Chandapura", "Begur",
        "Kengeri", "Uttarahalli", "Ramamurthy Nagar",
    ]

    location_lower = location.lower()

    for premium in premium_areas:
        if premium.lower() in location_lower:
            return "premium"

    for emerging in emerging_areas:
        if emerging.lower() in location_lower:
            return "emerging"

    return "good"


def generate_single_description(row: pd.Series, rng: random.Random) -> str:
    """
    Generate a natural-language property description from a single row.

    Args:
        row: A row from the Bengaluru housing DataFrame.
        rng: Seeded random.Random instance for reproducibility.

    Returns:
        A 2-4 sentence property description string.
    """
    # Extract features
    size = int(row.get("size", 2))
    total_sqft = row.get("total_sqft", 1000)
    bath = int(row.get("bath", 2))
    balcony = int(row.get("balcony", 1))
    location = row.get("location", "Bengaluru")
    price = row.get("price", 5000000)

    # Classify features
    area_class = _classify_area_size(total_sqft)
    price_class = _classify_price_segment(price)
    location_tier = _classify_location_tier(location)

    # Build the description sentence by sentence
    sentences = []

    # Sentence 1: BHK + property type + location
    bhk_desc = rng.choice(BHK_DESCRIPTIONS.get(size, BHK_DESCRIPTIONS[3]))
    prop_type = rng.choice(PROPERTY_TYPES)
    location_template = rng.choice(LOCATION_DESCRIPTIONS[location_tier])
    location_phrase = location_template.format(location=location)
    sentences.append(f"This {bhk_desc} {prop_type} is {location_phrase}.")

    # Sentence 2: Area + bathrooms + balcony
    area_quality = rng.choice(AREA_SIZE_DESCRIPTIONS[area_class])
    bath_phrase = rng.choice(BATHROOM_PHRASES).format(bath=bath)

    if balcony > 0:
        balcony_text = "balconies" if balcony > 1 else "balcony"
        balcony_phrase = rng.choice(BALCONY_PHRASES).format(balcony=balcony)
        sentences.append(f"The property features {area_quality}, {bath_phrase} and {balcony_phrase}.")
    else:
        sentences.append(f"The property features {area_quality} with {bath_phrase}.")

    # Sentence 3: Amenities + connectivity/IT proximity
    selected_amenities = rng.sample(AMENITIES, min(3, len(AMENITIES)))
    amenity_str = ", ".join(selected_amenities)

    # Choose between IT hub proximity or transport connectivity
    if rng.random() > 0.5:
        connectivity = rng.choice(IT_HUB_PROXIMITY)
    else:
        connectivity = rng.choice(TRANSPORT_CONNECTIVITY)

    quality_desc = rng.choice(BUILDING_QUALITY[price_class])
    sentences.append(f"Amenities include {amenity_str}, with {quality_desc} and {connectivity}.")

    # Sentence 4: Closing (70% chance)
    if rng.random() > 0.3:
        sentences.append(rng.choice(CLOSING_PHRASES))

    return " ".join(sentences)


def generate_descriptions(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate property descriptions for the entire DataFrame.

    Args:
        df: Housing DataFrame with Bengaluru real estate features.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with added 'property_description' column.
    """
    logger.info(f"Generating property descriptions for {len(df)} listings (seed={seed})")

    rng = random.Random(seed)
    df = df.copy()

    descriptions = []
    for idx, row in df.iterrows():
        desc = generate_single_description(row, rng)
        descriptions.append(desc)

    df["property_description"] = descriptions

    logger.info(f"Generated {len(descriptions)} property descriptions")
    logger.info(f"Avg description length: {df['property_description'].str.len().mean():.0f} chars")

    return df


if __name__ == "__main__":
    # Test generator with sample Bengaluru data
    from src.data.bengaluru_loader import load_bengaluru_housing

    df = load_bengaluru_housing()
    df_with_desc = generate_descriptions(df.head(10), seed=42)

    print("\n=== Sample Generated Descriptions ===\n")
    for idx, row in df_with_desc.iterrows():
        print(f"Location: {row['location']}, {row['size']} BHK, {row['total_sqft']} sqft")
        print(f"Description: {row['property_description']}\n")
