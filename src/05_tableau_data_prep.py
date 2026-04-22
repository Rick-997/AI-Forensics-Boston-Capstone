
# ========================================================
# 05 - Tableau Data Preparation
# AI Forensic Triage Tool — Predicting Shooting Incidents in Boston
# Goal: Load model + data from GitHub, generate predictions + SHAP values,
#       and export a clean CSV for Tableau dashboard
# ========================================================

import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb
import shap
import requests
from io import BytesIO
import tempfile
import os

print("✅ Starting Tableau Data Preparation (Notebook 05)")

# ====================== ROBUST PATHS (Spyder compatible) ======================
CURRENT_DIR = Path(__file__).parent.parent          # Go up to repo root
TABLEAU_FOLDER = CURRENT_DIR / "visualizations" / "tableau"
TABLEAU_FOLDER.mkdir(parents=True, exist_ok=True)

print(f"✅ Output folder ready → {TABLEAU_FOLDER.resolve()}")

# ================== RAW GITHUB URLS ==================
BASE = "https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main"
DATA_URL = f"{BASE}/data/processed/crimes_cleaned.parquet"
MODEL_URL = f"{BASE}/models/xgboost_shooting_model.json"

# ================== LOAD DATA ==================
r_data = requests.get(DATA_URL)
r_data.raise_for_status()
df = pd.read_parquet(BytesIO(r_data.content))
print(f"✅ Data loaded — {len(df):,} rows")

# ================== ONE-HOT ENCODE DISTRICT (exact match for model) ==================
district_dummies = pd.get_dummies(df['DISTRICT'], prefix='district')

expected_districts = [
    'district_A15', 'district_A7', 'district_B2', 'district_B3',
    'district_C11', 'district_C6', 'district_D14', 'district_D4',
    'district_E13', 'district_E18', 'district_E5',
    'district_External', 'district_Outside of', 'district_Unknown'
]

for col in expected_districts:
    if col not in district_dummies.columns:
        district_dummies[col] = 0

df = pd.concat([df, district_dummies[expected_districts]], axis=1)

# Add placeholder poverty_rate (model expects this column)
df['poverty_rate'] = 0.20
print("✅ DISTRICT one-hot encoded + poverty_rate added")

# ================== LOAD MODEL ==================
r_model = requests.get(MODEL_URL)
r_model.raise_for_status()

with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
    tmp.write(r_model.content)
    tmp_path = tmp.name

model = xgb.Booster()
model.load_model(tmp_path)
os.unlink(tmp_path)

print("✅ XGBoost model loaded successfully from GitHub")

# ================== GENERATE PREDICTIONS & SHAP ==================
feature_cols = ['hour', 'is_night', 'is_weekend', 'is_violent', 'poverty_rate'] + expected_districts

X = df[feature_cols]
df['predicted_prob'] = model.predict(xgb.DMatrix(X))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

for i, col in enumerate(feature_cols[:8]):
    df[f'shap_{col}'] = shap_values[:, i]

print("✅ Predictions + top SHAP values added")

# ================== SAVE FOR TABLEAU ==================
output_file = TABLEAU_FOLDER / "tableau_ready.csv"
df.to_csv(output_file, index=False)

print("\n🎉 SUCCESS! Tableau-ready file created")
print(f"   Location: {output_file}")
print(f"   Rows    : {len(df):,}")
print(f"   Columns : {len(df.columns)}")
print("\nYou can now open this CSV directly in Tableau!")