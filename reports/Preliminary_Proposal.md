# Preliminary Project Proposal

**Course**: DSE 6311 – Capstone  
**Submission Date**: March 2026  
**Author**: Ricardo Orellana (Solo)

## Basics
- **Who is Team Lead this week?** Ricardo Orellana  
- **Who is Recorder?** Ricardo Orellana  
- **Who is Spokesperson?** Ricardo Orellana  
- **Preliminary project title**: AI Forensic Triage Tool: Predicting Shooting Incidents in Boston to Prioritize Crime Lab Resources

## Background & Question
**Defined research question that serves a need or fills a niche**  
Can incident features (time of day, location, district, offense type proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?

**What is the question?**  
The question asks whether readily available public data can forecast shooting involvement in real time.

**What need or niche does it fill?**  
Forensic crime labs and the Boston Police Department process thousands of incidents under tight resource constraints. Ballistics, DNA, and firearms analysis are extremely time-intensive, yet not every reported crime involves a shooting. This tool fills the niche of real-time forensic triage so high-impact cases can be prioritized.

**Why is it worth your time/effort to explore this question?**  
It directly supports the Massachusetts State Police Crime Laboratory and Boston Police Department by reducing backlogs and improving public safety response times. The project uses only free public data and meets every capstone requirement while delivering immediate stakeholder value.

**Is your question novel / original?**  
It is not entirely novel (predictive policing exists), but applying it specifically to forensic lab triage with SHAP explainability for a real-world stakeholder in Boston is original and actionable.

**Identified stakeholder**  
Massachusetts State Police Crime Laboratory + Boston Police Department (they will use the model outputs to prioritize ballistics/DNA processing).

**Hypothesis and prediction**  
Hypothesis: Nighttime incidents in higher-poverty districts will show ≥35% higher predicted probability of involving a shooting.  
Prediction: The XGBoost model with SHAP explainability will achieve Precision-Recall AUC > 0.80 and clearly rank poverty_rate, hour, and is_night as the top drivers.

## Data & Analysis
**Data set(s) and why they are a good match**  
- Boston Police Crime Incident Reports (2023–present): ~150k rows, direct link: https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system  
- U.S. Census ACS 2020–2024 (Boston neighborhoods): poverty, income, education, race, housing density from https://data.census.gov  

These datasets are public, free, large enough for robust modeling, and perfectly aligned with the research question (incident-level features + neighborhood context).

**Response / outcome variable**  
SHOOTING (binary: 1 = shooting involved, 0 = no shooting)

**Predictor variables**  
Time features (hour, is_night, is_weekend), DISTRICT (one-hot), violent offense proxy, and district-level poverty_rate from Census ACS.

**Tentative analysis plan**  
1. Data cleaning & feature engineering (Notebook 01)  
2. Exploratory analysis and visualizations (Notebook 02)  
3. Modeling with XGBoost + SMOTE for imbalance + SHAP explainability (Notebook 03)  
4. Model evaluation and dependence plots (Notebook 04)  

Pitfalls I can already see: severe class imbalance (only ~0.7% shootings) and using district-level (not tract-level) poverty data. These will be addressed with SMOTE and noted as a limitation for future work.

**How will you know if your question is answered?**  
If the final model achieves PR-AUC > 0.80 and SHAP plots clearly show time and poverty as dominant predictors, the question is answered.

**How will you know if your hypothesis is supported?**  
If SHAP dependence plots and feature importance confirm that nighttime + high-poverty districts significantly increase shooting probability, the hypothesis is supported.

## Technical Details
- **Language I plan to code in**: Python (Jupyter Notebooks)  
- **Other resources needed**: None beyond free public datasets and standard libraries (pandas, scikit-learn, XGBoost, SHAP)  
- **Link to GitHub repo**: https://github.com/Rick-997/AI-Forensics-Boston-Capstone

---