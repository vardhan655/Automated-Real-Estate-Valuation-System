"""File I/O utilities for saving and loading artifacts."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info(f"Saved DataFrame ({df.shape[0]} rows, {df.shape[1]} cols) → {path}")


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from CSV."""
    df = pd.read_csv(path)
    logger.info(f"Loaded DataFrame ({df.shape[0]} rows, {df.shape[1]} cols) ← {path}")
    return df


def save_pickle(obj: Any, path: Path) -> None:
    """Save any Python object with joblib (handles sklearn pipelines, numpy arrays)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved artifact → {path}")


def load_pickle(path: Path) -> Any:
    """Load a joblib-serialized object."""
    obj = joblib.load(path)
    logger.info(f"Loaded artifact ← {path}")
    return obj


def save_numpy(arr: np.ndarray, path: Path) -> None:
    """Save a numpy array to .npy format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logger.info(f"Saved numpy array {arr.shape} → {path}")


def load_numpy(path: Path) -> np.ndarray:
    """Load a numpy array from .npy format."""
    arr = np.load(path)
    logger.info(f"Loaded numpy array {arr.shape} ← {path}")
    return arr


def save_json(data: Dict, path: Path) -> None:
    """Save a dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON → {path}")


def load_json(path: Path) -> Dict:
    """Load a JSON file as a dictionary."""
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON ← {path}")
    return data


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """Save evaluation metrics as formatted JSON."""
    # Round floats for readability
    rounded = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
    save_json(rounded, path)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist. Returns the path for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path
