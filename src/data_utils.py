# src/data_utils.py
from pathlib import Path

def get_repo_root():
    """Return the root path of the repository."""
    return Path(__file__).parents[1]

def load_data(filename="crime_features.csv"):
    """Load processed data from data/processed/."""
    path = get_repo_root() / "data" / "processed" / filename
    return pd.read_csv(path)