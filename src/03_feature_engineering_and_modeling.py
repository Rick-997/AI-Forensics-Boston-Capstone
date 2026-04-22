# 03 - Feature Engineering, SMOTE & Baseline Modeling

# ========================================================
# Full Test Script for Spyder - Notebook 03 Logic
# ========================================================

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import create_time_features, create_violent_flag
from src.data_utils import load_data

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("✅ Successfully imported custom functions from src/")

# 1. Load data
df = load_data("crime_features.csv")
print(f"Loaded {df.shape[0]:,} rows")
print(f"Initial shooting rate: {df['SHOOTING'].mean():.4%}")
print("Columns:", df.columns.tolist())

# 2. Feature Engineering using custom functions
df = create_time_features(df)
df = create_violent_flag(df)

print("✅ Feature engineering completed using custom functions")

# 3. Poverty rate
poverty_map = {
    'B3': 0.28, 'B2': 0.25, 'C11': 0.24, 'E18': 0.22, 'Unknown': 0.20,
    'E13': 0.21, 'E5': 0.18, 'A15': 0.17, 'C6': 0.16, 'D4': 0.15,
    'A7': 0.14, 'D14': 0.13, 'A1': 0.12, 'External': 0.10, 'Outside of': 0.10
}
df['poverty_rate'] = df['DISTRICT'].map(poverty_map).fillna(0.18)
print("✅ Poverty rate merged")

# 4. One-hot encoding
df = pd.get_dummies(df, columns=['DISTRICT'], prefix='district', drop_first=True)

feature_cols = ['hour', 'is_night', 'is_weekend', 'is_violent', 'poverty_rate'] + \
               [col for col in df.columns if col.startswith('district_')]

X = df[feature_cols]
y = df['SHOOTING']

print(f"Final feature matrix shape: {X.shape}")

print("\n🎉 Spyder test completed successfully!")