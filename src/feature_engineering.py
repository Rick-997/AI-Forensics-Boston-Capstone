# src/feature_engineering.py
import pandas as pd

def create_time_features(df):
    """Create circular time features and night/weekend flags."""
    df = df.copy()
    df['hour'] = pd.to_datetime(df['OCCURRED_ON_DATE']).dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['is_night'] = df['hour'].apply(lambda x: 1 if 20 <= x or x <= 6 else 0)
    df['is_weekend'] = pd.to_datetime(df['OCCURRED_ON_DATE']).dt.weekday >= 5
    return df

def create_violent_flag(df):
    """Create violent offense proxy."""
    violent_keywords = ['ASSAULT', 'ROBBERY', 'SHOOTING', 'HOMICIDE', 'KNIFE']
    df['is_violent'] = df['OFFENSE_CODE_GROUP'].astype(str).str.upper().str.contains('|'.join(violent_keywords))
    return df