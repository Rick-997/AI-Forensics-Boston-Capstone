# AI Forensic Triage Tool: Predicting Shooting Incidents in Boston

**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana (Team Sigma)

**[🔗 View Live Tableau Dashboard](https://public.tableau.com/app/profile/ricardo.orellana8607/viz/AIForensicTriageTool-BostonShootingRiskPredictor/AIForensicTriageToolBostonShootingRiskMap)**

---

## Project Title
AI Forensic Triage Tool: Predicting Shooting Incidents in Boston to Prioritize Crime Lab Resources

## Brief Background
Forensic crime labs face heavy caseloads and limited resources. Ballistics analysis, DNA testing, and firearms-related evidence are extremely time-intensive. However, not every reported crime involves a shooting. This project uses public Boston Police crime incident reports and U.S. Census neighborhood demographics to build an AI tool that predicts whether a reported incident will involve a shooting, allowing labs to prioritize high-impact cases and deliver faster justice.

## Research Question
Can incident features (time of day, location, district, offense type proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?

## Hypothesis & Predictions
Nighttime incidents in higher-poverty districts will show significantly higher probability of involving a shooting (expected ≥35% increase compared to daytime or lower-poverty areas).  
We predict the XGBoost model with SHAP explainability will achieve high Precision-Recall AUC and be ready for real-world use by the Massachusetts State Police Crime Laboratory and Boston Police Department.

## Objectives
- Predict shooting probability using time, location, district, and socioeconomic factors
- Achieve high Precision-Recall AUC despite severe class imbalance
- Provide SHAP explainability for real-world forensic use
- Deliver actionable insights for Massachusetts State Police Crime Laboratory and Boston Police Department

## Data Sources
- Boston Police Crime Incident Reports (2023–present): ~239k rows from [data.boston.gov](https://data.boston.gov)
- U.S. Census ACS 2020–2024: Boston neighborhoods (poverty, income, education, etc.)

## Methodology
- Data cleaning and feature engineering (circular time encoding, violent offense proxy, census merge)
- XGBoost classifier with class weighting (instead of SMOTE per professor feedback)
- Evaluation: Precision-Recall AUC, F1-score
- Interpretability: SHAP values + leakage test + fairness evaluation by night vs day
- Interactive visualization: Tableau dashboard

## Repository Structure
```bash
AI-Forensics-Boston-Capstone/
├── data/
│   ├── raw/
│   └── processed/
├── models/                          # XGBoost model + SHAP plots
├── notebooks/                       # Jupyter notebooks (for reference)
├── src/                             # Clean modular Python scripts
│   ├── 01_data_wrangling.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering_and_modeling.py
│   ├── 04_model_evaluation_and_shap.py
│   ├── 05_tableau_data_prep.py
│   ├── 06_hyperparameter_tuning_and_evaluation.py
│   ├── 07_final_model_leakage_test_and_fairness_evaluation.py
│   ├── data_utils.py
│   ├── feature_engineering.py
│   └── init.py
├── visualizations/
│   └── tableau/
├── reports/
├── final_delivery/
├── run_all.py                       # ← One-command full pipeline
├── README.md
└── requirements.txt
```

## How to Run the Full Pipeline

**Recommended (easiest way):**
```bash
python run_all.py

This single command runs all 7 scripts in the correct order.
```

## Team & Status

Solo project (Ricardo Orellana)
Complete: Data wrangling, EDA, modeling, SHAP explainability, leakage/fairness tests, and interactive Tableau dashboard

Last updated: April 2026