# Hyperparameter Tuning & Model Evaluation Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Prepared for**: Massachusetts State Police Crime Laboratory and Boston Police Department

### Background & Question (Recap)
Forensic crime laboratories face constant pressure to process evidence quickly while working with limited staff and equipment. Ballistics analysis, DNA testing, and firearm examinations are extremely time-consuming and resource-intensive. At the same time, the Boston Police Department responds to tens of thousands of crime incidents every year, but only a small fraction of those incidents actually involve a shooting. 

The current workflow treats every potential firearms-related report with the same level of urgency because there is no reliable, data-driven way to identify high-risk cases early. This creates significant backlogs and delays justice in the most serious cases. 

The central research question of this project is:  
**Can incident features (time of day, location, district, offense type proxies) combined with neighborhood demographics accurately predict whether a reported crime will involve a shooting?**

My hypothesis is that nighttime incidents in higher-poverty districts will show a significantly higher probability of involving a shooting — at least 35% higher than daytime or lower-poverty areas. In the previous phase I built a solid baseline model using XGBoost with class weighting. This report covers the hyperparameter tuning work I completed this week, the results obtained, and how they align with the overall project goals.

### Methods
I started from the baseline model developed in Notebook 03. In response to feedback received, I continued using **class weighting** (`scale_pos_weight = 141.59`) instead of SMOTE. This decision was made because class weighting trains the model on the original data distribution while giving more importance to the minority class, which feels more appropriate for real-world crime prediction where synthetic samples could introduce unrealistic patterns.

I performed hyperparameter tuning using `RandomizedSearchCV` with 20 random combinations and 3-fold cross-validation. The search focused on the most important XGBoost parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`. I kept `scale_pos_weight` fixed at the value calculated from the training set.

I used the same 80/20 stratified train/test split as before to ensure the test set reflected the real class imbalance. The primary evaluation metric remained Precision-Recall AUC because it is much more informative than accuracy when dealing with a rare positive class (only 0.70% shooting incidents).

All code for this phase is contained in the new Notebook 06. The entire project remains fully reproducible through the modular scripts in the `src/` folder and the master `run_all.py` script.

### Results & Brief Interpretations
The hyperparameter tuning process identified the following best parameters:
- `max_depth`: 4
- `learning_rate`: 0.05
- `n_estimators`: 200
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `scale_pos_weight`: 141.59

The tuned model achieved a **Test Precision-Recall AUC of 0.8377**. This is essentially identical to the baseline performance (0.8378), which suggests the original parameters were already quite strong. The more conservative `max_depth=4` and lower learning rate produced a slightly more stable model without sacrificing predictive power.

SHAP analysis on the tuned model confirmed the same key drivers seen in the baseline:
- `poverty_rate` remained the single most influential feature
- `is_night` and `hour` were the next strongest predictors
- Certain districts (particularly B2 and A7) continued to contribute noticeably

I generated new SHAP summary bar and beeswarm plots for the tuned model (saved as `shap_summary_bar_tuned.png` and `shap_summary_beeswarm_tuned.png`). These visualizations show very similar patterns to the baseline, which is reassuring and indicates the model is focusing on logical, interpretable features rather than noise.

Overall, the tuning experiment did not produce a large jump in performance, but it gave me greater confidence in the model’s stability and helped validate the feature engineering choices made earlier.

### Discussion & Next Steps
This week’s work reinforced that the baseline model was already performing well. The switch to class weighting continues to feel like the right approach for this type of imbalanced, real-world prediction task. The fact that the tuned model maintained almost identical performance with more conservative parameters suggests we have a reasonably robust starting point.

That said, several important areas still need attention in response to feedback received:
- I plan to run explicit tests for potential data leakage, especially from the engineered `is_violent` proxy and offense type features.
- Temporal and geographic structure in the data remains a concern. I will explore lagged time features and geography-aware validation splits to reduce the risk of the model simply learning “where and when” shootings tend to occur rather than the underlying drivers.
- Fairness is another priority. I intend to evaluate model performance stratified by district and by time of day to ensure we are not unintentionally reinforcing existing inequalities.
- The current use of district-level poverty rate from ACS data is somewhat coarse. In future iterations I would like to incorporate more granular neighborhood-level variables (education, income, housing density, etc.) to see if they improve the model.

**Next steps** before the final deliverable include:
- Comparing XGBoost against Random Forest and LightGBM to determine the best overall model
- Completing systematic hyperparameter tuning with a larger search space if time allows
- Conducting the fairness and leakage diagnostics mentioned above
- Adding more detailed SHAP dependence plots and individual prediction explanations
- Final integration of the best model into the interactive Tableau dashboard

I feel the project is progressing well. The combination of strong baseline performance, class weighting, and clear SHAP interpretability gives me confidence that the final tool will be both accurate and practically useful for the Massachusetts State Police Crime Laboratory and Boston Police Department.

**GitHub Repository**: https://github.com/Rick-997/AI-Forensics-Boston-Capstone  

**Live Tableau Dashboard**:  
[AI Forensic Triage Tool – Boston Shooting Risk Predictor](https://public.tableau.com/app/profile/ricardo.orellana8607/viz/AIForensicTriageTool-BostonShootingRiskPredictor/AIForensicTriageToolBostonShootingRiskMap)

**Last updated**: April 2026