# M01 Capstone Preliminary Proposal

**Project Title**: AI Forensic Triage Tool: Predicting Shooting Incidents in Boston to Prioritize Crime Lab Resources

**Author**: Ricardo Orellana 
**Course**: DSE 6311 – Capstone  
**Instructor**: Katherine Geist, Ph.D.

## Table of Contents
1. Introduction  
2. Problem Statement  
3. Data Source and Description  
4. Research Question and Hypothesis  
5. Variable Justification and Intersections  
6. Methodology  
   6.1 Data Cleaning & Preprocessing  
   6.2 Feature Engineering  
   6.3 Exploratory Data Analysis  
   6.4 Hypothesis Testing  
   6.5 Predictive Modeling  
7. Planned Visualization  
8. Expected Key Insights  
9. Target Audience/Stakeholder  
10. Conclusion  
References

## 1. Introduction
Forensic labs and police face heavy caseloads. Prioritizing incidents likely to involve shootings saves time on ballistics/DNA analysis.

## 2. Problem Statement
Most reported crimes do **not** involve shootings, yet evidence processing is uniform. A predictive model can triage high-risk cases.

## 3. Data Source and Description
- **Boston Police Crime Incident Reports (2023–present)**: ~150k+ rows CSV from data.boston.gov. Key columns: SHOOTING (target), OCCURRED_ON_DATE, DISTRICT, Lat/Long, OFFENSE_CODE_GROUP.
- **U.S. Census ACS 2020–2024**: Boston neighborhoods (poverty, income, education, race, housing density) — to be merged by district/zip.

## 4. Research Question and Hypothesis
**RQ**: Can time, location, district, and neighborhood demographics predict whether a reported crime involves a shooting?  
**Hypothesis**: Nighttime incidents in higher-poverty districts will show ≥35% higher predicted probability of involving a shooting.

## 5. Variable Justification and Intersections
- Time: hour_sin/cos, is_night, is_weekend  
- Location: DISTRICT (12 values), Lat/Long  
- Offense: violent proxy from OFFENSE_CODE_GROUP  
- Demographics (merge): poverty %, median income, education, race, housing density  
Interactions: night × high-poverty (key SHAP feature expected).

## 6. Methodology
6.1 Cleaning: Parse dates, fix missing Lat/Long/DISTRICT, binary SHOOTING target.  
6.2 Feature Engineering: Circular time encoding, district dummies, census merge, SMOTE for imbalance.  
6.3 EDA: Shooting rate by district/hour, correlations.  
6.4 Hypothesis Testing: Chi-square + preliminary logistic regression.  
6.5 Modeling: XGBoost/Random Forest classifier. Evaluation: Precision-Recall AUC, F1-score. SHAP explainability.

## 7. Planned Visualization
District heatmaps, SHAP summary plots, PR-curves, Tableau dashboard (final visuals only).

## 8. Expected Key Insights
Night + poverty will be top predictors; model >0.80 PR-AUC with clear SHAP interpretability.

## 9. Target Audience/Stakeholder
Massachusetts State Police Crime Laboratory + Boston Police Department (real triage tool).

## 10. Conclusion
Boston-specific, public-data project with direct public-safety impact.

## References
- Boston Police Data Portal  
- U.S. Census Bureau ACS  
- XGBoost & SHAP documentation