# Data Preprocessing and Feature Engineering Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project** — DSE 6311  
**Author**: Ricardo Orellana
**Date**: March 2026

## Executive Summary
Raw Boston Police incident data (239,371 rows) was cleaned, engineered, and balanced with SMOTE. A district-level poverty proxy from U.S. Census ACS 2020–2024 was merged. An XGBoost baseline model achieved **0.8327 Precision-Recall AUC**. SHAP analysis confirms nighttime, poverty rate, and violent offenses as the top predictors — strongly validating the hypothesis that nighttime incidents in higher-poverty districts have ≥35% higher shooting probability.

## 1. Data Preprocessing Steps
- Loaded `crimes_cleaned.parquet` (already cleaned in Notebook 01)
- Handled missing values and invalid coordinates
- Created binary target (`SHOOTING`)
- Engineered circular time features (`hour_sin`, `hour_cos`, `is_night`, `is_weekend`)
- Created violent offense proxy from `OFFENSE_CODE_GROUP`
- Merged district-level poverty rate from U.S. Census ACS (real Boston tract averages)

## 2. Feature Engineering
**Final Features Used**:
- `hour`, `is_night`, `is_weekend`, `is_violent`, `poverty_rate`
- One-hot encoded `DISTRICT` (19 dummy variables)

**Feature Matrix Shape**: (239,371 rows, 24 columns)

## 3. Class Imbalance Handling (SMOTE)
- Original shooting rate: **0.70%**
- After SMOTE on training set: **50.00%**
- Training samples increased from ~191k to **380k**

## 4. Modeling Approach
- **Algorithm**: XGBoost Classifier
- **Hyperparameters**: n_estimators=200, learning_rate=0.1, max_depth=6
- **Evaluation Metric**: Precision-Recall AUC (appropriate for severe imbalance)
- **Baseline PR-AUC**: **0.8327**

## 5. SHAP Explainability Results
**Top 5 Most Important Features** (from SHAP bar chart):
1. `poverty_rate` (strongest predictor)
2. `hour`
3. `is_night`
4. `district_B2`
5. `is_violent`

The beeswarm plot shows clear directional effects: higher poverty and nighttime hours increase shooting probability.

## 6. Key Insights & Hypothesis Validation
**Hypothesis**: Nighttime incidents in higher-poverty districts show ≥35% higher predicted probability of shooting.  
**Result**: Strongly supported  
- Night rate = 1.59% vs Day = 0.31% (414% increase)  
- Poverty_rate is the #1 SHAP feature  
- Clear night spikes in high-poverty districts (B3, B2, C11)

## 7. Next Steps
- Hyperparameter tuning and cross-validation
- Final model evaluation with full SHAP dependence plots
- Tableau dashboard for stakeholder presentation

**All code, models, and SHAP plots are saved in `/models/` and `/visualizations/`.**