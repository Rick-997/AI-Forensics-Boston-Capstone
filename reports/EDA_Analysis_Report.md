# Exploratory Data Analysis Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Stakeholder**: Massachusetts State Police Crime Laboratory and Boston Police Department  

---

## Recap Background & Question

Gunshot-related violence continues to be one of the most serious public safety challenges facing Boston. Every shooting incident requires extensive forensic resources, including ballistics analysis, firearm tracing, DNA testing on recovered evidence, and often multiple rounds of laboratory work by the Massachusetts State Police Crime Laboratory. These processes are time-intensive, expensive, and critical to building strong prosecutorial cases. However, the Boston Police Department receives tens of thousands of crime reports each year, and only a small fraction of them actually involve a shooting.  

The central problem is that current forensic workflows treat every reported incident with the same level of urgency because there is no reliable, data-driven method to identify which cases are most likely to involve firearms at the moment the incident is logged. This uniform approach creates significant backlogs, delays justice in the most serious cases, and places unnecessary strain on already limited laboratory resources.  

This Exploratory Data Analysis (EDA) directly addresses that operational gap. The primary research question is:  
**Can incident features (time of day, location, district, offense type proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?**  

The hypothesis is that nighttime incidents in higher-poverty districts will show significantly higher predicted probability of involving a shooting. The prediction is that these patterns will be strong enough to support an actionable triage model with SHAP explainability, allowing the crime laboratory to prioritize ballistics and DNA processing for the highest-risk cases. Understanding these patterns is essential for turning raw incident data into a practical decision-support tool that can accelerate justice while maintaining high recall of true shooting events.

---

## Methods

The dataset used for this EDA is the official Boston Police Crime Incident Reports covering 2023 to the present, containing 239,371 records. The data was acquired directly from the City of Boston’s open data portal[](https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system) and loaded via a public GitHub raw URL in `01_data_wrangling.ipynb` to ensure full reproducibility for any teammate or future stakeholder.

Key preprocessing steps included parsing the `OCCURRED_ON_DATE` timestamp to extract the hour of day and create binary flags for `is_night` (defined as 8 PM to 6 AM) and `is_weekend`. A `is_violent` proxy was engineered using string matching on the `OFFENSE_CODE_GROUP` field, with missing values handled via `.fillna("")` to avoid errors. The `DISTRICT` field was one-hot encoded to prepare it for modeling. All processed data was saved in both Parquet format (for computational efficiency) and CSV format (for easy GitHub visibility and sharing) in the `data/processed/` folder.

Exploratory analysis was conducted entirely in `02_eda.ipynb` using pandas for summary statistics and matplotlib/seaborn for visualizations. Six key plots were generated and exported as high-resolution PNG files to the `visualizations/` folder. A correlation matrix and summary table were also exported as CSV files. No external APIs or additional datasets were incorporated at this stage, as the focus was on thoroughly understanding the structure, quality, and patterns within the primary Boston incident data.

All code is fully documented, version-controlled in the GitHub repository, and designed to run end-to-end from raw data to final visualizations with a single notebook execution.

---

## Results, Visualizations, and Brief Interpretations

### Overall Shooting Rate and Class Imbalance
The target variable `SHOOTING` exhibits extreme class imbalance, with only 1,679 shooting incidents (0.70%) out of 239,371 total records. This imbalance is typical in real-world crime data but has important implications for modeling: standard accuracy metrics would be misleading, and techniques such as SMOTE oversampling combined with Precision-Recall AUC will be required in the next phase.

![Shooting Imbalance](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/01_shooting_imbalance.png)

### Temporal Patterns
Shooting probability increases dramatically during nighttime hours. The rate rises to 1.59% between 8 PM and 6 AM, compared with only 0.31% during daytime hours. This means nighttime incidents are approximately 5.14 times more likely to involve a shooting. The hourly trend plot shows a clear peak between 8 PM and 2 AM, with a gradual decline through the early morning.

![Shooting by Hour](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/02_shooting_by_hour.png)  
![Night vs Day](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/03_night_vs_day.png)

These temporal patterns strongly support the hypothesis that time of day is a critical predictor and provide the crime laboratory with an immediate, simple triage rule: incidents reported at night warrant heightened forensic attention.

### Geographic Patterns by District
Shooting rates vary substantially across Boston’s police districts. The highest rates are concentrated in three districts: B3 (Mattapan) at 1.67%, B2 (Roxbury) at 1.25%, and C11 (Dorchester) at 1.12%. These districts consistently emerge as hotspots and align with areas known to have higher levels of concentrated disadvantage.

![District Shooting Rates](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/04_district_rates.png)

### Hour × District Interaction
The most revealing visualization is the hour-by-district heatmap. It shows pronounced nighttime spikes in the highest-risk districts (B3, B2, C11, and E18), with shooting rates reaching 6.5% or higher in specific hour-district combinations. This interaction effect confirms that the combination of nighttime and specific geographic locations produces the highest risk and directly supports the project’s core hypothesis.

![Hour-District Heatmap](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/05_hour_district_heatmap.png)

### Violent Offense Proxy and Correlations
Incidents flagged as violent through the engineered proxy have a shooting rate of 8.80%, compared with only 0.48% for non-violent incidents. The correlation matrix further confirms that `is_violent`, `is_night`, and district indicators are among the strongest correlates with the target variable.

![Correlation Matrix](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/06_correlation_matrix.png)

---

## Discussion & Next Steps

The EDA has provided clear, actionable insights that strongly validate the research hypothesis. Nighttime incidents are over five times more likely to involve a shooting, and this risk is further amplified in specific high-poverty districts such as B2, B3, and C11. The hour-by-district heatmap is particularly powerful because it reveals the precise combinations of time and location that produce the highest shooting probability. These findings give the Massachusetts State Police Crime Laboratory and Boston Police Department an evidence-based foundation for prioritizing forensic resources on the small subset of incidents that are most likely to involve firearms.

The violent offense proxy also emerged as a highly informative early indicator, reinforcing the value of simple rule-based flags in addition to the full predictive model. The severe class imbalance observed (0.70% positive class) was expected and will be addressed in modeling through SMOTE oversampling and the use of Precision-Recall AUC as the primary evaluation metric.

**Next steps** include merging U.S. Census ACS socioeconomic variables (starting with poverty rate) by district to enrich the feature set, completing feature engineering, applying SMOTE, and training the XGBoost model with full SHAP explainability. The final interactive Tableau dashboard will combine the risk map, SHAP panels, and incident drill-down to give stakeholders an operational triage tool they can use daily.

All code, plots, and data dictionaries are fully documented in the GitHub repository, ensuring complete reproducibility and transparency for the crime laboratory team.

---

## Appendix: Data Dictionary

**Boston Police Crime Incident Reports (main dataset)**

| Column                | Description                                      | Type      | Notes |
|-----------------------|--------------------------------------------------|-----------|-------|
| INCIDENT_NUMBER       | Unique incident ID                               | String    | - |
| OFFENSE_CODE_GROUP    | Broad offense category                           | String    | Used for `is_violent` proxy |
| DISTRICT              | Police district                                  | String    | One-hot encoded |
| SHOOTING (target)     | Binary: 1 = shooting involved                    | Integer   | 0.70% positive |
| OCCURRED_ON_DATE      | Timestamp                                        | Datetime  | Engineered into hour, is_night, etc. |
| Lat / Long            | Geographic coordinates                           | Float     | Used for validation |
| hour                  | Hour of day (0–23)                               | Integer   | - |
| is_night              | 1 = 8 PM–6 AM                                    | Integer   | - |
| is_weekend            | 1 = Saturday or Sunday                           | Integer   | - |
| is_violent            | Proxy from OFFENSE_CODE_GROUP                    | Integer   | - |

**Engineered features** and all visualizations are available in the `visualizations/` folder.

---

**End of Report**  
This EDA provides a solid, stakeholder-focused foundation for the predictive modeling phase and directly supports more efficient forensic triage for the Massachusetts State Police Crime Laboratory and Boston Police Department.