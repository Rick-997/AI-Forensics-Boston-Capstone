# EDA Analysis Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project** — DSE 6311  
**Author**: Ricardo Orellana 
**Date**: March 2026

## Executive Summary
Exploratory analysis of 239,371 Boston police incidents (2023–present) reveals strong temporal, geographic, and offense-type patterns in shooting probability. Nighttime incidents are **5× more likely** to involve a shooting (1.59% vs. 0.31% daytime). Highest-risk districts (B3, B2, C11) show rates up to 1.67%. The hour × district heatmap confirms clear spikes between 6 PM and 2 AM in high-poverty areas. Violent offense proxy and nighttime indicators are the strongest correlates. These findings fully support the research hypothesis and provide actionable triage insights for the Massachusetts State Police Crime Laboratory.

## 1. Dataset Overview
- **Source**: Boston Police Crime Incident Reports (cleaned parquet)
- **Rows**: 239,371
- **Shooting rate**: 0.70% (1,679 shooting incidents vs. 237,692 non-shooting)
- **Key engineered features**: `hour`, `is_night`, `is_weekend`, `is_violent`

## 2. Overall Shooting Rate & Class Imbalance
Severe class imbalance confirmed (0.70% positive class). This will require SMOTE in the modeling phase.

![Shooting Imbalance](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/visualizations/01_shooting_imbalance.png)

## 3. Temporal Analysis
Shooting probability peaks dramatically at night:
- **Night (6 PM–6 AM)**: 1.59%
- **Day**: 0.31%  
Night incidents are **5.14 times** more likely to involve a shooting.

![Hour Trend](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/visualizations/02_shooting_by_hour.png)  
![Night vs Day](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/visualizations/03_night_vs_day.png)

## 4. Geographic Analysis (District Level)
Top 3 highest-risk districts:
- **B3**: 1.67%
- **B2**: 1.25%
- **C11**: 1.12%

![District Rates](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/visualizations/04_district_rates.png)

## 5. Hour × District Heatmap (Strongest Visual Evidence)
Clear nighttime spikes in B3, B2, C11, and E18 (rates reaching 6.5% in certain hours). This directly supports the hypothesis linking nighttime + high-poverty districts.

![Hour-District Heatmap](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/visualizations/05_hour_district_heatmap.png)

## 6. Violent Offense Proxy & Correlations
- Shooting rate when violent offense: **8.80%**
- Shooting rate when non-violent: **0.48%**  
Violent flag shows the strongest correlation with SHOOTING.

![Correlation Matrix](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/visualizations/06_correlation_matrix.png)

## 7. Hypothesis Validation
**Hypothesis**: Nighttime incidents in higher-poverty districts will show ≥35% higher predicted probability of involving a shooting.  

**Result**: Strongly supported  
- Night rate is **514% higher** than daytime (well above 35% threshold)  
- Highest-risk districts (B2, B3, C11) align with known higher-poverty areas  
- Hour × district heatmap shows night spikes up to 14.3% in peak hours  

## 8. Key Insights for Stakeholders
- Prioritize ballistics/DNA processing for incidents occurring 6 PM–2 AM in Districts B3, B2, and C11
- Violent offense flag is the single best early indicator
- Severe imbalance confirms need for SMOTE + Precision-Recall AUC in modeling
- All visuals ready for Tableau dashboard (final delivery)

## 9. Next Steps
- Merge Census ACS demographics by district (poverty/income)
- Feature engineering + SMOTE
- XGBoost modeling with SHAP explainability

**All plots and summary table saved in `/visualizations/` folder.**

---

