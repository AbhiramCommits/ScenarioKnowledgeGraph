"""Scenario embeddings."""

from pathlib import Path

import numpy as np
import pandas as pd


def embed_scenarios(summary_df: pd.DataFrame) -> np.ndarray:
    """Embed scenarios from summary stats."""
    features = []

    for _, row in summary_df.iterrows():
        feat = [
            float(row.get("n_agents", 1)),
            float(row.get("duration_steps", 50)),
            float(row.get("min_ttc", 5.0)),
            float(row.get("max_speed", 15.0)),
        ]
        features.append(feat)

    features = np.array(features, dtype=np.float32)

    # Normalize
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    features = (features - mean) / std

    return features


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    """Save embeddings to disk."""
    np.save(path, embeddings)


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from disk."""
    return np.load(path)
