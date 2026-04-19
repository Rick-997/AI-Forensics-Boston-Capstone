# Baseline Models Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Prepared for**: Massachusetts State Police Crime Laboratory and Boston Police Department

## Background & Question (Recap)

Forensic crime labs are under constant pressure to process evidence quickly while dealing with limited resources. Not every reported crime involves a shooting, yet each potential firearms-related case requires time-intensive work like ballistics analysis and DNA testing. The central question of this project remains the same: Can incident features (time of day, location, district, offense type proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?  

We hypothesized that nighttime incidents in higher-poverty districts would show a significantly higher shooting probability (at least 35% higher than daytime or lower-poverty areas). This report covers the baseline modeling work completed so far, including the choices made, assumptions, results, and what comes next.

## Methods

After finishing preprocessing and feature engineering, I trained a baseline XGBoost classifier using the prepared dataset (239,371 rows × 24 features). I chose XGBoost because it handles tabular data well, works with imbalanced classes, and pairs naturally with SHAP for explainability — both of which are important for stakeholders in the crime lab.

Key steps included:
- Splitting the data into training and test sets (80/20) while preserving the original class distribution in the test set.
- Applying SMOTE only on the training set to address the severe imbalance (original shooting rate was only 0.70%).
- Training the XGBoost model with reasonable default hyperparameters (n_estimators=200, learning_rate=0.1, max_depth=6).
- Using Precision-Recall AUC as the main evaluation metric because accuracy would be misleading with such a rare event.

No hyperparameter tuning was performed yet — this is a true baseline to establish a starting point.

## Results & Brief Interpretations

The baseline XGBoost model achieved a **Precision-Recall AUC of 0.8327**. This is a strong result given the extreme class imbalance and shows that the engineered features (especially nighttime, poverty rate, and district) carry real predictive signal.

SHAP analysis on the baseline model revealed clear patterns:
- Poverty Rate was the single most influential feature.
- Hour of Day and Is Night were the next strongest predictors, confirming the strong temporal component of risk.
- Specific districts (notably B2 and A7) also contributed meaningfully.

The model is already identifying the kinds of incidents we expected to be high-risk, which gives confidence that the feature engineering choices were on the right track. I also checked for overfitting by comparing training and test performance; the gap was reasonable, suggesting the baseline is not severely overfitting at this stage.

## Discussion & Next Steps

This baseline model confirms that the data preparation work was effective and that the hypothesis has a solid foundation. The Precision-Recall AUC of 0.8327 is encouraging for a first attempt, and the SHAP results align well with both criminological expectations and practical knowledge from the field.

That said, there is still room to improve. The next steps are:
1. Systematic hyperparameter tuning using cross-validation.
2. Testing additional models (Random Forest, LightGBM) for comparison.
3. Full regularization and overfitting controls (early stopping, feature selection).
4. Generating more detailed SHAP dependence plots and force plots for stakeholder presentations.
5. Integrating the final model predictions into the Tableau dashboard for interactive exploration.

Overall, the baseline has given us a clear and promising starting point. I’m looking forward to refining the model further and delivering a tool that the Massachusetts State Police Crime Laboratory and Boston Police Department can actually use in daily operations.

---

**GitHub Repository**: https://github.com/Rick-997/AI-Forensics-Boston-Capstone  

**Live Tableau Dashboard**:  
[AI Forensic Triage Tool – Boston Shooting Risk Predictor](https://public.tableau.com/app/profile/ricardo.orellana8607/viz/AIForensicTriageTool-BostonShootingRiskPredictor/AIForensicTriageToolBostonShootingRiskMap)

**All code, models, and SHAP plots** are saved in the repository under `notebooks/`, `models/`, and `visualizations/`.

---

