
# AI Forensic Triage Tool: Predicting Shooting Incidents in Boston

**Capstone Project** — DSE 6311  
**Goal**: Build a predictive model that helps forensic labs and Boston Police prioritize ballistics/DNA evidence by forecasting which incidents are likely to involve a shooting.

## Research Question
Can time, location, district, and neighborhood demographics predict whether a reported crime will involve a shooting?

## Datasets (public & original)
- Boston Police Crime Incident Reports (2015–present) → https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system
- U.S. Census ACS 2020–2024 Boston neighborhoods (income, poverty, education, etc.)

## Modeling
- XGBoost / Random Forest classifier
- SHAP explainability
- Metrics: Precision-Recall AUC, F1-score

## Stakeholder
Massachusetts State Police Crime Laboratory + Boston Police Department

## How to Run
1. `pip install -r requirements.txt`
2. Run scripts in order: `01_data_wrangling.py` → `02_eda.py` → ...

## Team & Status
- Solo for now (open to teammates)
- Currently in Week 1: data acquisition & cleaning

Last updated: March 2026
