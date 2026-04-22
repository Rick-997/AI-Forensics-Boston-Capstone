# src/data_utils.py
from pathlib import Path
import pandas as pd
import numpy as np

def get_repo_root():
    """Return the root path of the repository no matter where the script is run from."""
    current = Path(__file__).resolve().parent   # src/ folder
    # Go up until we find the repo root (contains notebooks/, data/, models/, etc.)
    for _ in range(5):  
        if (current / "notebooks").exists() or (current / "data").exists():
            return current
        current = current.parent
    return current  # fallback

def load_data(filename="crime_features.csv"):
    """Load processed data from data/processed/"""
    path = get_repo_root() / "data" / "processed" / filename
    if filename.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)