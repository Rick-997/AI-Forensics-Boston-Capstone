# 01_data_wrangling.py
import pandas as pd
import numpy as np
from pathlib import Path

# ================== PATHS (GitHub-friendly + works from notebooks/) ==================
# Direct link to the raw CSV on GitHub (anyone on the team can run this)
RAW_CSV_URL = "https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/data/raw/crime_incident_reports_2023_present.csv"

# Robust path that always finds the repo root
CURRENT_DIR = Path.cwd()
if CURRENT_DIR.name == "src":
    REPO_ROOT = CURRENT_DIR.parent          # Go up one level from notebooks/
else:
    REPO_ROOT = CURRENT_DIR

DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print(f"✅ Files will be saved here: {DATA_PROCESSED.resolve()}")

# ================== 1. LOAD FROM GITHUB ==================
print("Loading Boston Crime CSV directly from GitHub...")
df = pd.read_csv(RAW_CSV_URL, parse_dates=["OCCURRED_ON_DATE"], low_memory=False)
print(f"Original shape: {df.shape}")

# ================== 2. TARGET (already 0/1) ==================
df["SHOOTING"] = df["SHOOTING"].astype(int)
print(f"Shooting rate: {df['SHOOTING'].mean():.4%} (severe imbalance — SMOTE next)")

# ================== 3. TIME FEATURES ==================
df["hour"] = df["OCCURRED_ON_DATE"].dt.hour
df["is_weekend"] = df["OCCURRED_ON_DATE"].dt.dayofweek.isin([5, 6]).astype(int)
df["is_night"] = ((df["hour"] >= 20) | (df["hour"] <= 5)).astype(int)

# Circular encoding (great for XGBoost)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ================== 4. LOCATION & CLEANING ==================
# Keep only valid Boston coordinates
df = df[(df["Lat"] > 42.2) & (df["Lat"] < 42.4) & 
        (df["Long"] > -71.2) & (df["Long"] < -71.0)]
df["DISTRICT"] = df["DISTRICT"].fillna("Unknown")

# Offense proxy — FIXED VERSION
df["is_violent"] = (df["OFFENSE_CODE_GROUP"]
                    .fillna("")                    # ← this prevents NaN crash
                    .astype(str)                   # ← forces string dtype
                    .str.contains("Assault|Robbery|Homicide|Murder", 
                                  case=False, na=False)
                    .astype(int))

# ================== 5. KEEP & SAVE ==================
cols_keep = [
    "INCIDENT_NUMBER", "OFFENSE_CODE_GROUP", "DISTRICT", "Lat", "Long",
    "SHOOTING", "hour", "is_weekend", "is_night", "hour_sin", "hour_cos",
    "is_violent", "YEAR", "MONTH"
]
df_clean = df[cols_keep].copy()

df_clean.to_parquet(DATA_PROCESSED / "crimes_cleaned.parquet", index=False)
df_clean.to_csv(DATA_PROCESSED / "crime_features.csv", index=False)

print(f"✅ Cleaned data saved! New shape: {df_clean.shape}")
print(df_clean["SHOOTING"].value_counts(normalize=True))