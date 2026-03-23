# Model Evaluation and Interpretation Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project** — DSE 6311  
**Author**: Ricardo Orellana
**Date**: March 2026

## Executive Summary
The final XGBoost model achieved a strong **0.8327 Precision-Recall AUC** on the test set. SHAP analysis confirms that poverty rate, hour of day, and nighttime are the dominant predictors. The dependence plot clearly shows that higher poverty combined with nighttime dramatically increases shooting probability — fully validating the research hypothesis and providing actionable triage guidance for the Massachusetts State Police Crime Laboratory.

## 1. Final Model Performance
- **Algorithm**: XGBoost Classifier
- **Test Set PR-AUC**: **0.8327**
- **Classification Report**:
  - Precision (Shooting = 1): 0.82
  - Recall (Shooting = 1): 0.76
  - F1-Score (Shooting = 1): 0.79
  - Accuracy: 0.99 (driven by imbalance, but PR-AUC is the key metric)

## 2. SHAP Feature Importance
**Top 5 Most Important Features**:
1. `poverty_rate` (strongest overall predictor)
2. `hour`
3. `is_night`
4. `district_B2`
5. `is_violent`

## 3. SHAP Dependence Plots (Key Interactions)
The Poverty Rate × Night dependence plot shows a clear positive interaction: as poverty rate increases, the SHAP value for shooting probability rises sharply during nighttime hours. This directly supports the hypothesis that nighttime incidents in higher-poverty districts have ≥35% higher predicted shooting probability.

![SHAP Dependence Plot](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/models/shap_dependence_poverty_night.png)

## 4. Interpretation for Stakeholders
- **Triage Rule**: Prioritize ballistics/DNA processing for any incident with high poverty_rate + nighttime (6 PM–6 AM) in districts B3, B2, and C11.
- **Actionable Insight**: The model can be deployed in real time to flag high-risk cases, reducing backlog and improving response times.
- **Limitations**: Severe class imbalance (0.70%) — future work could explore additional features or ensemble methods.

## 5. Next Steps
- Final Capstone Project Report with Tableau dashboard
- Model deployment recommendations for Boston Police / State Police Crime Lab

**All code, final model, and SHAP plots are saved in `/models/`.**

---

