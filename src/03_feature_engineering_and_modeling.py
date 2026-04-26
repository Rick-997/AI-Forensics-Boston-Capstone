# ========================================================
# 03 - Feature Engineering & Baseline Modeling
# AI Forensic Triage Tool — Predicting Shooting Incidents in Boston
# ========================================================

import sys
from pathlib import Path

# Robust path for running from inside src/ folder
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import create_time_features, create_violent_flag
from src.data_utils import load_data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("✅ Successfully imported custom functions from src/")

# ====================== 1. LOAD DATA ======================
df = load_data("crime_features.csv")
print(f"Loaded {df.shape[0]:,} rows")
print(f"Initial shooting rate: {df['SHOOTING'].mean():.4%}")

# ====================== 2. FEATURE ENGINEERING ======================
df = create_time_features(df)
df = create_violent_flag(df)

print("✅ Feature engineering completed using custom functions")

# ====================== 3. POVERTY RATE ======================
poverty_map = {
    'B3': 0.28, 'B2': 0.25, 'C11': 0.24, 'E18': 0.22, 'Unknown': 0.20,
    'E13': 0.21, 'E5': 0.18, 'A15': 0.17, 'C6': 0.16, 'D4': 0.15,
    'A7': 0.14, 'D14': 0.13, 'A1': 0.12, 'External': 0.10, 'Outside of': 0.10
}
df['poverty_rate'] = df['DISTRICT'].map(poverty_map).fillna(0.18)
print("✅ Poverty rate merged by district (ACS proxy)")

# ====================== 4. ONE-HOT ENCODING ======================
df = pd.get_dummies(df, columns=['DISTRICT'], prefix='district', drop_first=True)

feature_cols = ['hour', 'is_night', 'is_weekend', 'is_violent', 'poverty_rate'] + \
               [col for col in df.columns if col.startswith('district_')]

X = df[feature_cols]
y = df['SHOOTING']

print(f"Final feature matrix shape: {X.shape}")

# ====================== 5. TRAIN/TEST SPLIT ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate class weight (replaces SMOTE - per professor feedback)
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"✅ Calculated scale_pos_weight = {scale_pos_weight:.2f} (negative/positive ratio)")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Shooting rate in training set: {y_train.mean():.4%}")
print("✅ Using class weighting instead of SMOTE")

# ====================== 6. TRAIN XGBoost BASELINE ======================
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,      # Class weighting
    random_state=42,
    eval_metric='aucpr'
)

model.fit(X_train, y_train)
print("✅ XGBoost baseline model trained with class weighting")

# ====================== 7. EVALUATION ======================
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print("\n=== BASELINE MODEL PERFORMANCE ===")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(classification_report(y_test, y_pred))

# ====================== 8. SHAP EXPLAINABILITY ======================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Bar chart
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
ax.set_title("SHAP Feature Importance (Top 15)")
fig.tight_layout()
fig.savefig("../models/shap_summary_bar.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("✅ SHAP bar chart saved")

# Beeswarm plot
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
ax.set_title("SHAP Summary Plot (Beeswarm)")
fig.tight_layout()
fig.savefig("../models/shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("✅ SHAP beeswarm plot saved")

# Save model
model.save_model("../models/xgboost_shooting_model.json")
print("✅ Model + SHAP plots saved to models/ folder")

print("\n🎉 03_feature_engineering_and_modeling.py completed successfully!")