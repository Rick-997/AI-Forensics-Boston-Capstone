# Hyperparameter Tuning & Model Evaluation Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Prepared for**: Massachusetts State Police Crime Laboratory and Boston Police Department

### Background & Question (Recap)
The goal of this project is to help forensic labs and Boston Police quickly identify which reported incidents are most likely to involve a shooting so they can prioritize ballistics and DNA work.  

The main research question is still:  
**Can incident features (time of day, location, district, offense type proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?**

My hypothesis is that nighttime incidents in higher-poverty districts carry significantly higher risk. Last week I built a strong baseline using XGBoost with class weighting. This report covers the hyperparameter tuning I did this week and what I learned.

### Methods
I started from the baseline model I built in Notebook 03. To avoid the issues with SMOTE that the professor mentioned, I continued using **class weighting** (`scale_pos_weight = 141.59`) instead of synthetic samples.

I performed hyperparameter tuning with `RandomizedSearchCV` (20 random combinations, 3-fold CV) on the most important XGBoost parameters:
- n_estimators, max_depth, learning_rate, subsample, colsample_bytree

I kept the same 80/20 stratified train/test split and evaluated everything with Precision-Recall AUC. I did not try other models (Random Forest or LightGBM) yet — I wanted to focus on tuning XGBoost first.

All code is in the new Notebook 06 and the modular scripts in the `src/` folder.

### Results & Brief Interpretations
The tuned model achieved a **Test Precision-Recall AUC of 0.8377**.  
This is almost identical to the baseline (0.8378), which tells me the original parameters were already quite good.

**Best parameters found:**
- max_depth = 4
- learning_rate = 0.05
- n_estimators = 200
- subsample = 0.8
- colsample_bytree = 0.8
- scale_pos_weight = 141.59

SHAP analysis on the tuned model still shows the same top drivers:
- poverty_rate (strongest)
- is_night
- hour

The new SHAP bar chart and beeswarm plot (saved as `shap_summary_bar_tuned.png` and `shap_summary_beeswarm_tuned.png`) look very similar to the baseline, which is reassuring.

I did not see major overfitting. The model remains focused on logical, interpretable features.

### Discussion & Next Steps
The tuning experiment confirmed that the baseline was already strong. The more conservative `max_depth=4` and lower learning rate made the model slightly more stable without hurting performance much.

I’m glad I followed the professor’s advice and switched to class weighting — it feels much more appropriate for this real-world use case.

Things I still want to improve:
- Run a full comparison with Random Forest and LightGBM
- Do explicit leakage tests (especially around the `is_violent` proxy)
- Add more granular ACS neighborhood variables (education, income, housing) instead of just district-level poverty
- Run fairness checks across districts and time of day
- Explore lagged time features or geography-aware splits to better handle temporal and spatial structure

Next week I plan to pick the final “best model,” integrate it into the Tableau dashboard, and start preparing the final deliverable.

Overall, I feel the project is on a good track and the model is becoming more robust and trustworthy for the crime lab.

**GitHub Repository**: https://github.com/Rick-997/AI-Forensics-Boston-Capstone  

**Live Tableau Dashboard**:  
[AI Forensic Triage Tool – Boston Shooting Risk Predictor](https://public.tableau.com/app/profile/ricardo.orellana8607/viz/AIForensicTriageTool-BostonShootingRiskPredictor/AIForensicTriageToolBostonShootingRiskMap)

**Last updated**: April 2026