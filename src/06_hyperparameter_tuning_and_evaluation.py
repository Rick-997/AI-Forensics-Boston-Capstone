# =============================================================================
# 06 - Hyperparameter Tuning & Model Evaluation
# AI Forensic Triage Tool: Predicting Shooting Incidents in Boston
# Capstone Project — DSE 6311
# Author: Ricardo Orellana
# =============================================================================

import sys
from pathlib import Path

# Make src/ importable from anywhere
sys.path.append("..")

from src.feature_engineering import create_time_features, create_violent_flag
from src.data_utils import load_data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, auc
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("✅ Notebook 06 started - Hyperparameter Tuning & Model Evaluation")

# =============================================================================
# 2. Load Data & Recreate Features (same as baseline)
# =============================================================================
df = load_data("crime_features.csv")
print(f"Loaded {df.shape[0]:,} rows")
print(f"Initial shooting rate: {df['SHOOTING'].mean():.4%}")

# Feature engineering (safe for already-processed data)
df = create_time_features(df)
df = create_violent_flag(df)

# Poverty rate + one-hot encoding
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

print(f"Final feature matrix shape: {X.shape}")

# =============================================================================
# 3. Train/Test Split + Class Weighting (no SMOTE)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"✅ Calculated scale_pos_weight = {scale_pos_weight:.2f}")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Shooting rate in training set: {y_train.mean():.4%}")
print("✅ Using class weighting (recommended for real crime prediction)")

# =============================================================================
# 4. Hyperparameter Tuning with RandomizedSearchCV
# =============================================================================
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [scale_pos_weight]
}

random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(random_state=42, eval_metric='aucpr'),
    param_distributions=param_dist,
    n_iter=20,
    scoring='average_precision',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("🔄 Starting hyperparameter tuning (RandomizedSearchCV)...")
random_search.fit(X_train, y_train)

print(f"✅ Best parameters found: {random_search.best_params_}")
print(f"✅ Best cross-validation PR-AUC: {random_search.best_score_:.4f}")

# =============================================================================
# 5. Train Final Model with Best Parameters
# =============================================================================
best_params = random_search.best_params_

final_model = xgb.XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='aucpr'
)

final_model.fit(X_train, y_train)
print("✅ Final model trained with best hyperparameters")

# =============================================================================
# 6. Final Evaluation on Test Set
# =============================================================================
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print("\n=== FINAL TUNED MODEL PERFORMANCE ===")
print(f"Test Precision-Recall AUC: {pr_auc:.4f}")
print(classification_report(y_test, y_pred))

# =============================================================================
# 7. SHAP Analysis for Tuned Model
# =============================================================================
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

# Bar chart
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
ax.set_title("SHAP Feature Importance - Tuned Model (Top 15)")
fig.tight_layout()
fig.savefig("../models/shap_summary_bar_tuned.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("✅ Tuned SHAP bar chart saved")

# Beeswarm plot
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
ax.set_title("SHAP Summary Plot (Beeswarm) - Tuned Model")
fig.tight_layout()
fig.savefig("../models/shap_summary_beeswarm_tuned.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("✅ Tuned SHAP beeswarm plot saved")

print("\n🎉 Notebook 06 completed successfully!")