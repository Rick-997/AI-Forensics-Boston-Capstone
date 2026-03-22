# AI Forensic Triage Tool: Predicting Shooting Incidents in Boston

**Capstone Project** — DSE 6311  
**Goal**: Build a predictive model that helps forensic labs and Boston Police prioritize ballistics/DNA evidence by forecasting which incidents are likely to involve a shooting.

## Overview
Forensic crime labs face heavy caseloads. This project uses public Boston crime data and neighborhood demographics to predict whether a reported incident involves a shooting, enabling smarter triage and faster justice.

## Objectives
- Predict shooting probability using time, location, district, and socioeconomic factors
- Achieve high Precision-Recall AUC despite severe class imbalance
- Provide SHAP explainability for real-world forensic use
- Deliver actionable insights for Massachusetts State Police Crime Laboratory and Boston Police Department

## Data Sources
- **Boston Police Crime Incident Reports (2023–present)**: ~150k rows from https://data.boston.gov (SHOOTING = target)
- **U.S. Census ACS 2020–2024**: Boston neighborhoods (poverty, income, education, race, housing density)

## Methodology
- Data cleaning and feature engineering (time circular encoding, violent proxy, census merge)
- XGBoost / Random Forest classifier with SMOTE
- Evaluation: Precision-Recall AUC, F1-score
- Interpretability: SHAP values

## Repository Organization
- `data/raw/` — original downloads
- `data/processed/` — cleaned files
- `notebooks/` — numbered analysis scripts
- `reports/` — all deliverables
- `src/` — reusable Python code
- `models/` — saved models
- `visualizations/` — plots and Tableau exports

## How to Run
1. `pip install -r requirements.txt`
2. Run notebooks in order: `01_data_wrangling.ipynb` → `02_eda.ipynb` → …
3. All outputs saved automatically to `data/processed/`

## Team & Status
- Solo (Ricardo) — open to teammates if anyone joins
- Currently in Week 1: data acquisition & cleaning complete

Last updated: March 2026