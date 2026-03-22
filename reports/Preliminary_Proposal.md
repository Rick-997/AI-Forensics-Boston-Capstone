# Preliminary Proposal

**Project Title**: AI Forensic Triage Tool: Predicting Shooting Incidents in Boston to Prioritize Crime Lab Resources

**Author**: Ricardo Orellana
**Course**: DSE 6311 – Capstone  
**Instructor**: Katherine Geist, Ph.D.  
**Date**: March 2026

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
Forensic crime laboratories and police departments process thousands of incidents annually under tight resource constraints. Ballistics analysis, DNA testing, and firearms-related evidence are time-intensive. However, not every reported crime involves a shooting. This project develops an AI Forensic Triage Tool that predicts shooting probability in real time using public Boston data, allowing labs to prioritize high-impact cases and improve public safety outcomes.

## 2. Problem Statement
Current evidence-processing workflows treat all incidents uniformly, creating backlogs and delaying justice in shooting-related cases. With over 300,000 incidents in the Boston dataset since 2015, manual triage is inefficient. A predictive model using incident features and neighborhood demographics can flag high-probability shooting cases for faster forensic analysis.

## 3. Data Source and Description
- **Boston Police Crime Incident Reports (2023–present)**: ~150k rows CSV from https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system. Key columns: SHOOTING (binary target), OCCURRED_ON_DATE, DISTRICT, Lat/Long, OFFENSE_CODE_GROUP.
- **U.S. Census ACS 2020–2024**: Boston neighborhoods/tracts (median income, poverty rate, education, race, housing density) from data.census.gov. These will be merged by district to add socioeconomic context.

Both datasets are public, free, and require original cleaning — fulfilling capstone requirements.

## 4. Research Question and Hypothesis
**Research Question**: Can incident features (time of day, location, district, offense proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?  

**Hypothesis**: Nighttime incidents in higher-poverty districts will show ≥35% higher predicted probability of involving a shooting.

## 5. Variable Justification and Intersections
- Time features (hour_sin/cos, is_night, is_weekend) capture known violent-crime patterns.
- Location (DISTRICT + Lat/Long) reflects Boston’s geographic disparities.
- Offense proxies (violent flag) serve as strong predictors.
- Census demographics (poverty, income, education, race) add the social determinants layer.  
Key interactions: night × high-poverty (expected top SHAP feature).

## 6. Methodology
6.1 Data Cleaning & Preprocessing: Parse dates, handle missing values, filter invalid coordinates, create binary SHOOTING target.  
6.2 Feature Engineering: Circular time encoding, violent proxy, district dummies, census merge, SMOTE for class imbalance.  
6.3 Exploratory Data Analysis: Shooting rates by district/hour, correlation matrices.  
6.4 Hypothesis Testing: Chi-square tests and preliminary logistic regression.  
6.5 Predictive Modeling: XGBoost / Random Forest classifier. Evaluation: Precision-Recall AUC, F1-score. SHAP explainability.

## 7. Planned Visualization
District heatmaps, time-of-day line plots, SHAP summary and dependence plots, and a final Tableau dashboard for stakeholder presentation.

## 8. Expected Key Insights
Nighttime and poverty will emerge as dominant predictors. The model is expected to achieve >0.80 PR-AUC with clear, actionable SHAP explanations for forensic triage.

## 9. Target Audience/Stakeholder
Massachusetts State Police Crime Laboratory and Boston Police Department — direct users for real-world case prioritization and resource allocation.

## 10. Conclusion
This Boston-specific, public-data project delivers a practical AI tool with immediate public-safety value while meeting all capstone requirements.

## References
- Boston Police Data Portal (data.boston.gov)  
- U.S. Census Bureau American Community Survey  
- XGBoost and SHAP documentation  
- Past capstone examples (IOTA team)
