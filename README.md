# AI Forensic Triage Tool: Predicting Shooting Incidents in Boston

**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  

## Interactive Dashboard
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

## Overview
Forensic crime labs face heavy caseloads. This project uses public Boston crime data and neighborhood demographics to predict whether a reported incident involves a shooting, enabling smarter triage and faster justice.

## Objectives
- Predict shooting probability using time, location, district, and socioeconomic factors
- Achieve high Precision-Recall AUC despite severe class imbalance
- Provide SHAP explainability for real-world forensic use
- Deliver actionable insights for Massachusetts State Police Crime Laboratory and Boston Police Department

## Data Sources
- Boston Police Crime Incident Reports (2023–present): ~239k rows from [data.boston.gov](https://data.boston.gov) (SHOOTING = target)
- U.S. Census ACS 2020–2024: Boston neighborhoods (poverty, income, education, race, housing density)

## Methodology
- Data cleaning and feature engineering (circular time encoding, violent offense proxy, census merge)
- XGBoost classifier with SMOTE for class imbalance
- Evaluation: Precision-Recall AUC, F1-score
- Interpretability: SHAP values
- Interactive visualization: Tableau dashboard

## Repository Organization
- `data/raw/` — original downloads
- `data/processed/` — cleaned files
- `notebooks/` — numbered analysis scripts (01–05)
- `reports/` — all deliverables
- `models/` — saved XGBoost model
- `visualizations/` — plots and Tableau workbook
- `visualizations/tableau/` — packaged `.twbx` dashboard file

## How to Run
1. `pip install -r requirements.txt`
2. Run notebooks in order: `01_data_wrangling.ipynb` → `02_eda.ipynb` → `03_feature_engineering_and_modeling.ipynb` → `04_model_evaluation_and_shap.ipynb` → `05_tableau_data_prep.ipynb`
3. Open `AI_Forensic_Triage_Tool_Boston_Shooting_Risk_Predictor.twbx` in Tableau Desktop/Public for the interactive dashboard

## Team & Status
- Solo project (Ricardo Orellana)
- **Complete**: Data wrangling, EDA, modeling, SHAP explainability, and interactive Tableau dashboard published

**Last updated**: April 2026