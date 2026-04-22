# ========================================================
# 04 - Model Evaluation and Interpretation
# AI Forensic Triage Tool — Predicting Shooting Incidents in Boston
# ========================================================

import sys
from pathlib import Path

# Robust path detection 
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("✅ Starting Model Evaluation & SHAP Interpretation")

# ====================== 1. ROBUST PATHS ======================
CURRENT_DIR = Path(__file__).parent.parent
MODELS = CURRENT_DIR / "models"
DATA_PROCESSED = CURRENT_DIR / "data" / "processed"

MODELS.mkdir(parents=True, exist_ok=True)
print(f"✅ Models folder ready → {MODELS.resolve()}")

# ====================== 2. LOAD MODEL + RECREATE FEATURES ======================
model = xgb.XGBClassifier()
model.load_model(MODELS / "xgboost_shooting_model.json")
print("✅ XGBoost model loaded successfully")

# Load the cleaned data and recreate exact same features as Notebook 03
df = pd.read_parquet(DATA_PROCESSED / "crimes_cleaned.parquet")

poverty_map = {
    'B3': 0.28, 'B2': 0.25, 'C11': 0.24, 'E18': 0.22, 'Unknown': 0.20,
    'E13': 0.21, 'E5': 0.18, 'A15': 0.17, 'C6': 0.16, 'D4': 0.15,
    'A7': 0.14, 'D14': 0.13, 'A1': 0.12, 'External': 0.10, 'Outside of': 0.10
}

df['poverty_rate'] = df['DISTRICT'].map(poverty_map).fillna(0.18)
df = pd.get_dummies(df, columns=['DISTRICT'], prefix='district', drop_first=True)

feature_cols = ['hour', 'is_night', 'is_weekend', 'is_violent', 'poverty_rate'] + \
               [col for col in df.columns if col.startswith('district_')]

X = df[feature_cols]
y = df['SHOOTING']

print(f"✅ Features recreated - Shape: {X.shape}")

# ====================== 3. FINAL MODEL PERFORMANCE ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print("\n=== FINAL MODEL PERFORMANCE ===")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(classification_report(y_test, y_pred))

# ====================== 4. SHAP DEPENDENCE PLOTS ======================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Poverty × Night interaction (most important one)
shap.dependence_plot(
    "poverty_rate", 
    shap_values, 
    X_test, 
    interaction_index="is_night", 
    show=False
)
plt.title("SHAP Dependence: Poverty Rate × Night")
plt.tight_layout()
plt.savefig(MODELS / "shap_dependence_poverty_night.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP Dependence plot saved: shap_dependence_poverty_night.png")

print("\n🎉 Notebook 04 completed successfully!")
print("All outputs saved in the models/ folder")