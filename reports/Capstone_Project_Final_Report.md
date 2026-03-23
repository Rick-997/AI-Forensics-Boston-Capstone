# Capstone Project Final Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston to Prioritize Crime Lab Resources**

**DSE 6311 Capstone Project**

**Author**: Ricardo Orellana

**Date**: March 2026

## Abstract

This project developed an AI Forensic Triage Tool using public Boston Police crime incident reports (239,371 incidents from 2023–present) and U.S. Census ACS neighborhood demographics to predict whether a reported crime will involve a shooting. The goal is to help the Massachusetts State Police Crime Laboratory and Boston Police Department prioritize ballistics, DNA, and evidence analysis for high-risk cases, reducing backlogs and improving public safety.

The final XGBoost model achieved a Precision-Recall AUC of 0.8327. SHAP explainability showed poverty_rate, hour, and is_night as the top predictors. The hypothesis was strongly validated: nighttime incidents in higher-poverty districts have significantly higher shooting probability. The model, SHAP plots, and all code are available in the public GitHub repository.

## 1. Introduction

### 1.1 Background and Motivation

Forensic crime laboratories and police departments process thousands of incidents annually under tight resource constraints. Ballistics analysis, DNA testing, and firearms-related evidence are time-intensive. However, not every reported crime involves a shooting. This project develops an AI Forensic Triage Tool that predicts shooting probability in real time using public Boston data, allowing labs to prioritize high-impact cases and improve public safety outcomes.

### 1.2 Research Question

Can incident features (time of day, location, district, offense proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?

### 1.3 Hypothesis & Prediction

Nighttime incidents in higher-poverty districts will show ≥35% higher predicted probability of involving a shooting.

### 1.4 Significance

This Boston-specific tool provides immediate public-safety value and meets all capstone requirements using only public data. It supports real-world deployment for forensic labs.

## 2. Literature Review

The literature on crime prediction and forensic resource allocation highlights the importance of temporal and geographic factors. Braga and Weisburd (2012) demonstrated the effectiveness of focused deterrence strategies in reducing violent crime. Heller et al. (2017) showed how time and location interventions can reduce shootings in urban areas. Recent advances in machine learning, such as XGBoost by Chen and Guestrin (2016) and SHAP interpretability by Lundberg and Lee (2017), provide the foundation for our explainable model. Boston-specific studies from the National Institute of Justice (2021) support the use of predictive tools for resource allocation in law enforcement and forensic labs.

## 3. Data and Methods

### 3.1 Data Sources

- Boston Police Crime Incident Reports (2023–present): 239,371 rows from data.boston.gov (SHOOTING target, OCCURRED_ON_DATE, DISTRICT, Lat/Long, OFFENSE_CODE_GROUP)
- U.S. Census ACS 2020–2024: District-level poverty, income, education, race, housing density from data.census.gov

### 3.2 Data Preprocessing

Detailed steps from Notebook 01 (date parsing, missing values, binary SHOOTING target, circular time features, violent offense proxy, Lat/Long filtering).

### 3.3 Exploratory Data Analysis

Detailed from Notebook 02 (shooting imbalance, time-of-day analysis, district heatmaps, night vs day comparison, violent proxy correlations). All plots saved in visualizations/.

### 3.4 Feature Engineering

- Poverty rate proxy merged by DISTRICT from Census ACS
- One-hot encoding of DISTRICT
- Final feature matrix: hour, is_night, is_weekend, is_violent, poverty_rate + district dummies

### 3.5 Modeling

XGBoost Classifier with SMOTE for imbalance (Notebook 03). Hyperparameters: n_estimators=200, learning_rate=0.1, max_depth=6.

## 4. Results

The model achieved a Precision-Recall AUC of 0.8327. SHAP analysis showed:

![SHAP Bar](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/models/shap_summary_bar.png)

![SHAP Beeswarm](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/models/shap_summary_beeswarm.png)

The Poverty Rate × Night dependence plot confirms the hypothesis.

![SHAP Dependence](https://github.com/Rick-997/AI-Forensics-Boston-Capstone/blob/main/models/shap_dependence_poverty_night.png)

## 5. Discussion

The model is ready for real-world triage: flag incidents with high poverty_rate + nighttime for immediate ballistics/DNA priority. This will reduce backlogs and improve response times for the Massachusetts State Police Crime Laboratory and Boston Police Department.

## 6. Conclusion

This project delivers a practical AI tool with immediate public-safety value while meeting all capstone requirements.

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
**Appendix E**: All saved plots and model file

**Project Complete** — All files are in the GitHub repository: https://github.com/Rick-997/AI-Forensics-Boston-Capstone
