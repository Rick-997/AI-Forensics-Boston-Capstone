# Capstone Project Final Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston to Prioritize Crime Lab Resources**

**DSE 6311 Capstone Project**  
**Author**: Ricardo Orellana 
**Date**: March 2026

## Abstract

Forensic crime laboratories and police departments face heavy caseloads and limited resources. Ballistics analysis, DNA testing, and firearms-related evidence require significant time and specialized equipment, yet not every reported crime involves a shooting. This project developed an AI Forensic Triage Tool using public Boston Police crime incident reports (239,371 incidents from 2023–present) and U.S. Census ACS neighborhood demographics to predict whether a reported incident will involve a shooting. The goal is to help the Massachusetts State Police Crime Laboratory and Boston Police Department prioritize high-impact cases for faster evidence processing.

The final XGBoost model achieved a strong Precision-Recall AUC of 0.8327 despite severe class imbalance. SHAP explainability identified poverty_rate, hour, and is_night as the dominant predictors. The hypothesis was strongly validated: nighttime incidents in higher-poverty districts show dramatically higher shooting probability (night rate 1.59% vs day 0.31%, a 414% increase). The model, SHAP plots, dependence analysis, and all code are available in the public GitHub repository and ready for real-world deployment.

## 1. Introduction

### 1.1 Background and Motivation

Forensic crime laboratories and police departments process thousands of incidents annually under tight resource constraints. Ballistics analysis, DNA testing, and firearms-related evidence are time-intensive and require specialized equipment. However, not every reported crime involves a shooting. Current workflows treat all incidents uniformly, leading to backlogs and delayed justice in shooting-related cases. With over 300,000 incidents in the Boston dataset since 2015, manual triage is inefficient.

A data-driven predictive model using incident features and neighborhood demographics can flag high-probability shooting cases in real time, enabling forensic labs to allocate resources strategically and improve response times for public safety. This project addresses a critical gap in forensic resource allocation by developing a Boston-specific AI triage tool that is explainable, reproducible, and ready for deployment.

### 1.2 Research Question

Can incident features (time of day, location, district, offense type proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?

### 1.3 Hypothesis & Prediction

Nighttime incidents in higher-poverty districts will show ≥35% higher predicted probability of involving a shooting. We expect the model to achieve >0.80 Precision-Recall AUC with clear, actionable SHAP explanations.

### 1.4 Significance

This project delivers a practical AI tool with immediate public-safety value. It uses only free public data, requires original cleaning and feature engineering, and directly supports the Massachusetts State Police Crime Laboratory and Boston Police Department in prioritizing ballistics and DNA analysis. The work meets all DSE 6311 capstone requirements and can serve as a template for other cities.

## 2. Literature Review

The literature on crime prediction and forensic resource allocation highlights the importance of temporal, geographic, and socioeconomic factors. Braga and Weisburd (2012) demonstrated that focused deterrence strategies targeting high-risk locations and times significantly reduce violent crime. Heller et al. (2017) showed how time-of-day and location-based interventions can reduce shootings in urban areas. The National Institute of Justice (2021) emphasizes the role of predictive policing tools in law enforcement resource allocation.

Recent advances in machine learning provide the technical foundation for this project. Chen and Guestrin (2016) introduced XGBoost, a scalable tree boosting system that excels with imbalanced data. Lundberg and Lee (2017) developed SHAP values for model interpretability, which is essential for forensic and law enforcement stakeholders who need to understand why a case is flagged as high-risk. Weisburd et al. (2016) highlighted the geographic concentration of crime ("place matters"), supporting our district-level and poverty-rate analysis.

These studies collectively justify our choice of XGBoost with SHAP and the inclusion of nighttime and socioeconomic features.

## 3. Data and Methods

### 3.1 Data Sources

- **Boston Police Crime Incident Reports (2023–present)**: 239,371 rows from https://data.boston.gov. Key columns: SHOOTING (binary target), OCCURRED_ON_DATE, DISTRICT, Lat/Long, OFFENSE_CODE_GROUP.
- **U.S. Census ACS 2020–2024**: Boston neighborhoods/tracts (median income, poverty rate, education, race, housing density) from https://data.census.gov. Merged by DISTRICT to add socioeconomic context.

### 3.2 Data Preprocessing

Detailed steps from Notebook 01:
- Parsed dates and created binary SHOOTING target
- Engineered circular time features (hour_sin/cos, is_night, is_weekend)
- Created violent offense proxy from OFFENSE_CODE_GROUP
- Filtered invalid coordinates and handled missing values

### 3.3 Exploratory Data Analysis

Detailed in Notebook 02:
- Severe class imbalance confirmed (0.70% shooting cases)
- Night shooting rate 1.59% vs day 0.31% (5× higher)
- Highest-risk districts: B3 (1.67%), B2 (1.25%), C11 (1.12%)
- Hour × district heatmap showed clear nighttime spikes

**Figure 3.1 Shooting Imbalance**  
![Shooting Imbalance](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/01_shooting_imbalance.png)

**Figure 3.2 Shooting Rate by Hour**  
![Shooting by Hour](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/02_shooting_by_hour.png)

### 3.4 Feature Engineering

- Merged district-level poverty_rate proxy from Census ACS
- One-hot encoded DISTRICT (19 dummy variables)
- Final feature matrix: hour, is_night, is_weekend, is_violent, poverty_rate + district dummies (24 columns)

### 3.5 Modeling

- Algorithm: XGBoost Classifier
- Imbalance handling: SMOTE on training set (50% shooting rate after balancing)
- Hyperparameters: n_estimators=200, learning_rate=0.1, max_depth=6
- Evaluation metric: Precision-Recall AUC

## 6. Results

The final model achieved a Precision-Recall AUC of **0.8327**.

**Figure 6.1 SHAP Feature Importance (Top 15)**  
![SHAP Feature Importance](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/models/shap_summary_bar.png)

**Figure 6.2 SHAP Summary Plot (Beeswarm)**  
![SHAP Summary Beeswarm](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/models/shap_summary_beeswarm.png)

**Figure 6.3 SHAP Dependence: Poverty Rate × Night**  
![SHAP Dependence Poverty x Night](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/models/shap_dependence_poverty_night.png)

The plots confirm that higher poverty combined with nighttime dramatically increases shooting probability.

## 7. Discussion

The model is ready for real-world triage: flag incidents with high poverty_rate + nighttime (6 PM–6 AM) in Districts B3, B2, and C11 for immediate ballistics/DNA priority. This will reduce backlogs and improve response times. The SHAP dependence plot provides clear, explainable evidence for stakeholders.

Limitations include severe class imbalance (addressed with SMOTE) and the use of a district-level poverty proxy (future work could use tract-level data).

## 8. Conclusion

This Boston-specific AI Forensic Triage Tool delivers immediate public-safety value while meeting all capstone requirements. The model, plots, and reports are ready for deployment.

## References

1. Boston Police Department. Crime Incident Reports. https://data.boston.gov  
2. U.S. Census Bureau. American Community Survey. https://data.census.gov  
3. Chen, T., & Guestrin, C. (2016). XGBoost. KDD Conference.  
4. Lundberg, S. M., & Lee, S.-I. (2017). SHAP. NeurIPS.  
5. Braga, A. A., & Weisburd, D. (2012). Focused Deterrence. Journal of Research in Crime and Delinquency.  
6. Heller, S. B., et al. (2017). Crime Reduction in Chicago. Quarterly Journal of Economics.  
7. National Institute of Justice. (2021). Predictive Policing.  
8. Weisburd, D., et al. (2016). Place Matters. Cambridge University Press.

## Appendices

**Appendix A**: Full data preprocessing code (Notebook 01)  
**Appendix B**: EDA plots and code (Notebook 02)  
**Appendix C**: Feature engineering and modeling code (Notebook 03)  
**Appendix D**: Model evaluation and SHAP code (Notebook 04)  
**Appendix E**: All saved plots and model file (`xgboost_shooting_model.json`)

**Project Complete** — All files are in the GitHub repository: https://github.com/Rick-997/AI-Forensics-Boston-Capstone