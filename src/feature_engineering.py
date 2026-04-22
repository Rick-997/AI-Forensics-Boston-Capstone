# src/feature_engineering.py
import pandas as pd
import numpy as np

def create_time_features(df):
    """Create circular time features and night/weekend flags.
    Works whether the original OCCURRED_ON_DATE exists or 'hour' is already present."""
    df = df.copy()
    
    if 'OCCURRED_ON_DATE' in df.columns:
        print("✅ Creating time features from OCCURRED_ON_DATE")
        df['hour'] = pd.to_datetime(df['OCCURRED_ON_DATE']).dt.hour
    elif 'hour' in df.columns:
        print("✅ 'hour' column already exists - skipping time feature creation")
    else:
        raise KeyError("Neither 'OCCURRED_ON_DATE' nor 'hour' column found in the data.")

    # Circular encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Binary flags
    df['is_night'] = df['hour'].apply(lambda x: 1 if (x >= 20 or x <= 6) else 0)
    
    # Weekend flag (safe fallback)
    if 'OCCURRED_ON_DATE' in df.columns:
        df['is_weekend'] = pd.to_datetime(df['OCCURRED_ON_DATE']).dt.weekday >= 5
    else:
        df['is_weekend'] = 0   # fallback if date column is missing

    return df


def create_violent_flag(df):
    """Create violent offense proxy."""
    df = df.copy()
    violent_keywords = ['ASSAULT', 'ROBBERY', 'SHOOTING', 'HOMICIDE', 'KNIFE', 'VIOLENT']
    if 'OFFENSE_CODE_GROUP' in df.columns:
        df['is_violent'] = df['OFFENSE_CODE_GROUP'].astype(str).str.upper().str.contains('|'.join(violent_keywords))
    else:
        df['is_violent'] = 0
    return df