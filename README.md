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
│   ├── 08_model_comparison_leakage_mitigation_fairness.py
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
```bash
# 1. Clone the repo
git clone https://github.com/Rick-997/AI-Forensics-Boston-Capstone.git
cd AI-Forensics-Boston-Capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run everything with one command 
python src/run_all.py        			# ← This single command runs all 8 scripts in the correct order.
```

## Testing & Reproducibility

The complete end-to-end pipeline can be executed with a single command:

```bash
python src/run_all.py
```
This script runs all eight modules in the correct order:

Data wrangling → EDA → Feature engineering → Modeling → Hyperparameter tuning → Final evaluation → Leakage & fairness checks → Tableau data preparation

Key features that ensure full reproducibility:

- All code is modular and stored as clean .py files in the src/ folder
- Custom functions are centralized in src/data_utils.py and src/feature_engineering.py
- Dependencies are pinned in requirements.txt
- The pipeline has been tested on a clean environment and consistently produces the same outputs (models/, visualizations/, and tableau_ready.csv)

The full final report is available here:
reports/Project_Final_Report.md


## Team & Status

Team Sigma (Ricardo Orellana)
Complete end-to-end pipeline: (Data Wrangling → EDA → Modeling → Hyperparameter Tuning → Leakage & Fairness Testing → Tableau)

Last updated: May 2026