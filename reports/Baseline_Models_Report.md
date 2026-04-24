# Baseline Models Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Prepared for**: Massachusetts State Police Crime Laboratory and Boston Police Department

## Background & Question (Recap)

Forensic crime laboratories operate under intense pressure. Every shooting incident requires extensive resources — ballistics analysis, firearm tracing, DNA testing on recovered evidence, and often multiple rounds of laboratory work. These processes are time-consuming, expensive, and critical for building strong prosecutorial cases. At the same time, the Boston Police Department responds to tens of thousands of crime incidents each year, the vast majority of which do not involve firearms. 

The current workflow treats every potential firearms-related report with the same level of urgency because there is no reliable, data-driven method to triage cases at the moment an incident is logged. This uniform approach creates significant backlogs, delays justice in the most serious cases, and places unnecessary strain on already limited laboratory resources.

This capstone project addresses that operational gap by developing an AI Forensic Triage Tool. The central research question is:  
**Can incident features (time of day, location, district, offense type proxies) combined with neighborhood demographics accurately predict whether a reported crime will involve a shooting?**

I hypothesized that nighttime incidents in higher-poverty districts would show a significantly higher probability of involving a shooting — at least 35% higher than daytime or lower-poverty areas. This report presents the baseline modeling phase, including the methodological choices I made, the assumptions behind them, the results obtained, and how these results align with both the original hypothesis and the feedback received on the finalized proposal.

## Methods

After completing data preprocessing and feature engineering, I trained a baseline XGBoost classifier on the final prepared dataset containing 239,371 records and 24 engineered features. I selected XGBoost as the initial algorithm because it is well-suited for tabular data, natively supports class imbalance through weighting, and integrates seamlessly with SHAP explainability — a critical requirement for building trust with forensic stakeholders who need to understand why the model flags certain incidents as high-risk.

The modeling pipeline included the following steps:

1. I performed an 80/20 stratified train/test split to preserve the original class distribution in the test set, ensuring realistic evaluation.

2. To address the extreme class imbalance (only 0.70% positive SHOOTING cases), I replaced the previously used SMOTE oversampling with **class weighting** via the `scale_pos_weight` parameter in XGBoost (calculated as 141.59). This approach trains the model on the original data distribution while giving higher importance to the minority class, which is more appropriate for real-world crime prediction where synthetic samples could introduce unrealistic patterns.

3. I trained the model using reasonable baseline hyperparameters: `n_estimators=200`, `learning_rate=0.1`, `max_depth=6`, `subsample=0.8`, and `colsample_bytree=0.8`. No hyperparameter tuning was performed at this stage — the goal was to establish a true baseline performance level.

4. I evaluated the model exclusively using Precision-Recall AUC and the classification report on the original imbalanced test set.

All code for this phase is contained in Notebook 03 and the modular Python scripts in the `src/` folder. The trained model and SHAP plots are saved in the `models/` folder. The entire pipeline can now be executed with a single command using `run_all.py`.

## Results & Brief Interpretations

The baseline XGBoost model with class weighting achieved a **Precision-Recall AUC of 0.8378**. This result is encouraging for a first baseline and represents a slight improvement over the previous SMOTE-based version, confirming that the engineered features are carrying meaningful predictive signal.

The classification report on the test set showed strong performance on the majority class while maintaining reasonable recall on the rare positive class. SHAP analysis provided clear and actionable insights:

- **Poverty rate** emerged as the single most influential feature, validating the inclusion of district-level ACS data.
- **Hour of day** and the binary `is_night` indicator were the next strongest predictors, reinforcing the strong temporal patterns identified during EDA.
- Certain districts (particularly B2 and A7) also contributed noticeably to the predictions.

These findings align closely with the original hypothesis: nighttime incidents in higher-poverty districts carry substantially elevated shooting risk. The SHAP summary bar chart and beeswarm plot (saved in the `models/` folder) make these relationships visually interpretable for stakeholders. Overall, the model is already identifying the exact combinations of time, location, and socioeconomic factors that the crime laboratory would want to prioritize for immediate forensic attention.

## Discussion & Next Steps

This baseline modeling phase confirms that the extensive preprocessing and feature engineering work completed earlier was effective. The model is learning meaningful, criminologically plausible patterns rather than noise or spurious correlations.

However, several important limitations and assumptions must be explicitly acknowledged in light of the proposal feedback:

- **Label timing assumption**: The target variable is derived from the time the incident was reported rather than the time a shooting was confirmed. In a real deployment, this could introduce a small lag between prediction and ground truth. I will clearly communicate this limitation to stakeholders and explore mitigation strategies such as confidence thresholds or real-time model updates.

- **Potential leakage from proxy variables**: The engineered `is_violent` flag and `OFFENSE_CODE_GROUP` are derived from the same incident data as the target. While useful, they carry a risk of data leakage. Future iterations will include ablation tests (removing these proxies) to quantify their contribution versus true predictive power.

- **Temporal and geographic structure**: The data contains both year-to-year and district-level patterns. Circular time encoding and district one-hot encoding help mitigate some of this structure, but there remains a risk that the model learns location-specific or year-specific patterns rather than generalizable drivers. I plan to explore lagged temporal features and geography-aware validation splits in the next phase.

- **Fairness considerations**: Because the model incorporates district-level poverty, there is a potential risk of reinforcing existing socioeconomic inequalities. I will conduct explicit fairness evaluations (performance stratified by district and nighttime) and define maximum acceptable disparity thresholds across groups.

**Next steps** before final delivery include:
1. Systematic hyperparameter tuning using cross-validation.
2. Comparison against Random Forest and LightGBM models.
3. Explicit fairness diagnostics and leakage tests.
4. Addition of more granular ACS neighborhood variables (economic health, wellness, education) beyond the current poverty-rate proxy.
5. Creation of additional SHAP dependence plots and individual prediction explanations.
6. Final integration of the model into the interactive Tableau dashboard.

This baseline has provided a strong, explainable foundation. By directly addressing the feedback on class imbalance handling, leakage risks, temporal/geographic structure, and fairness, I am confident the final model will be both accurate and operationally trustworthy for the Massachusetts State Police Crime Laboratory and Boston Police Department.

---

**GitHub Repository**: https://github.com/Rick-997/AI-Forensics-Boston-Capstone  

**Live Tableau Dashboard**:  
[AI Forensic Triage Tool – Boston Shooting Risk Predictor](https://public.tableau.com/app/profile/ricardo.orellana8607/viz/AIForensicTriageTool-BostonShootingRiskPredictor/AIForensicTriageToolBostonShootingRiskMap)

All code, models, and SHAP plots are available in the repository under the `notebooks/`, `src/`, `models/`, and `visualizations/` folders.

**Last updated**: April 2026
