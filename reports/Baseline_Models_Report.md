# Baseline Models Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Prepared for**: Massachusetts State Police Crime Laboratory and Boston Police Department

## Background & Question (Recap)

Forensic crime laboratories face constant pressure to process evidence quickly while working with limited staff and equipment. Ballistics analysis, DNA testing, and firearm examinations are extremely time-consuming. In Boston, the police department responds to tens of thousands of incidents each year, yet only a small fraction actually involve a shooting. This creates a real operational bottleneck: the crime lab must decide which cases deserve immediate attention.

The central research question of this project is:  
**Can incident features (time of day, location, district, offense type proxies) combined with neighborhood demographics accurately predict whether a reported crime will involve a shooting?**

I hypothesized that nighttime incidents in higher-poverty districts would show a significantly higher shooting probability. This report presents the baseline modeling results, the modeling choices I made, the assumptions behind them, and how I plan to address the limitations identified in the proposal feedback.

## Methods

After completing preprocessing and feature engineering, I trained a baseline XGBoost classifier on the final dataset (239,371 rows). I chose XGBoost because it performs well on tabular data, handles class imbalance natively through weighting, and integrates seamlessly with SHAP for explainability — a key requirement for stakeholder trust.

Key modeling steps:
- I performed an 80/20 stratified train/test split to preserve the original class distribution in the test set.
- To address the severe class imbalance (only 0.70% positive SHOOTING cases), I replaced SMOTE with **class weighting** using `scale_pos_weight = 141.59`. This approach avoids creating synthetic samples and is more appropriate for real-world crime prediction.
- I trained the model on the original (non-resampled) training data using the following hyperparameters: `n_estimators=200`, `learning_rate=0.1`, `max_depth=6`.
- I used Precision-Recall AUC as the primary evaluation metric and kept evaluation on the original imbalanced test set.

All code is contained in Notebook 03 and the modular `src/` scripts. The trained model is saved in the `models/` folder.

## Results & Brief Interpretations

The baseline XGBoost model achieved a **Precision-Recall AUC of 0.8378**. This is a solid result for the first baseline and shows that the engineered features carry meaningful predictive signal without relying on synthetic data.

SHAP analysis revealed clear patterns:
- Poverty rate was the single most influential feature.
- Hour of day and the `is_night` indicator were the next strongest predictors, confirming the strong temporal signal identified in EDA.
- Certain districts (particularly B2 and A7) also contributed noticeably.

These findings align well with the original hypothesis: nighttime incidents in higher-poverty districts carry substantially elevated shooting risk. The model is already identifying the types of cases the crime lab would want to prioritize. I observed reasonable generalization between training and test performance, suggesting the baseline is not severely overfitting at this stage.

I generated SHAP summary bar and beeswarm plots (saved in the `models/` folder) to support interpretability for stakeholders.

## Discussion & Next Steps

This baseline confirms that the feature engineering work was effective and that the model is learning meaningful patterns rather than noise. The switch to class weighting improved realism while maintaining strong performance.

However, several important limitations and assumptions must be acknowledged:

- **Label timing assumption**: The target variable is based on the time the incident was reported, not the time a shooting was confirmed. In a real deployment, this could create a small lag between prediction and ground truth. I will clearly communicate this limitation to stakeholders and explore ways to mitigate it (e.g., real-time updates or confidence thresholds).
- **Potential leakage from proxy variables**: The `is_violent` flag and `OFFENSE_CODE_GROUP` are derived from the same incident data as the target. While useful, they carry a risk of data leakage. I plan to test model performance with and without these proxies to quantify their impact.
- **Temporal and geographic structure**: The data has both year-to-year and district-level structure. Circular time encoding and district one-hot encoding help, but I recognize the risk of the model learning location-specific or year-specific patterns instead of generalizable drivers. In future work I will explore lagged temporal features and geography-aware validation splits.
- **Fairness considerations**: Because the model uses district-level poverty, there is a risk of reinforcing existing inequalities. I will conduct explicit fairness checks (e.g., performance stratified by district and nighttime) and set a maximum acceptable disparity threshold across groups.

**Next steps** (before final delivery):
1. Systematic hyperparameter tuning and comparison with Random Forest / LightGBM.
2. Explicit fairness evaluation and leakage diagnostics.
3. Additional SHAP dependence plots and individual prediction explanations.
4. Integration of more granular ACS neighborhood variables (economic health, wellness, education) beyond the current poverty-rate proxy.
5. Final deployment of the interactive Tableau dashboard.

This baseline has given me a strong, explainable foundation. I am confident that addressing the points above will produce a model that is both accurate and trustworthy for operational use by the Massachusetts State Police Crime Laboratory and Boston Police Department.

---

**GitHub Repository**: https://github.com/Rick-997/AI-Forensics-Boston-Capstone  

**Live Tableau Dashboard**:  
[AI Forensic Triage Tool – Boston Shooting Risk Predictor](https://public.tableau.com/app/profile/ricardo.orellana8607/viz/AIForensicTriageTool-BostonShootingRiskPredictor/AIForensicTriageToolBostonShootingRiskMap)

All code, models, and SHAP plots are available in the repository.

---

**Last updated**: April 2026

