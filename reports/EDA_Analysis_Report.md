# Exploratory Data Analysis Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Stakeholder**: Massachusetts State Police Crime Laboratory and Boston Police Department  

---

## Recap Background & Question

The core research question for this project is:  
**Can incident features (time of day, location, district, offense type proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?**

This question addresses a critical operational need for forensic laboratories. Every shooting incident triggers intensive ballistics, DNA, and firearms-related analysis. However, with over 239,000 reported incidents in Boston since 2023 and only ~0.7% involving a shooting, current workflows treat every case with equal urgency. This creates backlogs and delays justice in the highest-impact cases. The goal of this EDA is to uncover temporal, geographic, and socioeconomic patterns that can help the crime laboratory prioritize resources more effectively.  

The hypothesis is that nighttime incidents in higher-poverty districts will show significantly elevated shooting probability. The prediction is that these patterns will be strong enough to support an actionable triage model with SHAP explainability for forensic stakeholders.

---

## Methods

The dataset used is the Boston Police Crime Incident Reports (2023–present), containing 239,371 records. The data was acquired directly from the official public portal[](https://data.boston.gov) and loaded via a GitHub-friendly URL in `01_data_wrangling.ipynb`.  

Key preprocessing steps included:  
- Parsing `OCCURRED_ON_DATE` to extract hour, weekday, and binary `is_night` (8 PM–6 AM) and `is_weekend` flags.  
- Creating a `is_violent` proxy using string matching on `OFFENSE_CODE_GROUP`.  
- One-hot encoding the `DISTRICT` field.  
- Saving both Parquet (for speed) and CSV (for visibility) versions in the `data/processed/` folder.  

Exploratory analysis was performed in `02_eda.ipynb`. All visualizations were generated with matplotlib/seaborn and saved as high-resolution PNGs in the `visualizations/` folder. Summary statistics and correlation matrices were exported as CSV for easy review. No external APIs or additional datasets were needed at this stage.

---

## Results, Visualizations, and Brief Interpretations

### Overall Shooting Rate and Class Imbalance
The target variable `SHOOTING` shows extreme imbalance: only **1,679 shooting incidents** (0.70%) out of 239,371 total records. This confirms the need for SMOTE oversampling and Precision-Recall AUC in the modeling phase.

![Shooting Imbalance](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/01_shooting_imbalance.png)

### Temporal Patterns
Shooting probability increases dramatically at night:  
- **Night (8 PM–6 AM)**: 1.59%  
- **Day**: 0.31%  

Night incidents are **5.14 times** more likely to involve a shooting. The hourly trend peaks sharply between 8 PM and 2 AM.

![Shooting by Hour](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/02_shooting_by_hour.png)  
![Night vs Day](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/03_night_vs_day.png)

### Geographic Patterns by District
The highest shooting rates are concentrated in three districts:  
- **B3 (Mattapan)**: 1.67%  
- **B2 (Roxbury)**: 1.25%  
- **C11 (Dorchester)**: 1.12%  

These districts consistently show elevated risk and align with known higher-poverty areas.

![District Shooting Rates](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/04_district_rates.png)

### Hour × District Interaction (Key Insight)
The heatmap reveals clear nighttime spikes in B3, B2, and C11, with rates reaching **6.5%+** in peak hours. This interaction strongly supports the hypothesis that nighttime + specific districts is the highest-risk combination for forensic triage.

![Hour-District Heatmap](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/05_hour_district_heatmap.png)

### Violent Offense Proxy and Correlations
Incidents flagged as violent have a shooting rate of **8.80%** (vs. 0.48% for non-violent). The correlation matrix confirms that `is_violent`, `is_night`, and district indicators are the strongest correlates with the target.

![Correlation Matrix](https://raw.githubusercontent.com/Rick-997/AI-Forensics-Boston-Capstone/main/visualizations/06_correlation_matrix.png)

---

## Discussion & Next Steps

The EDA clearly validates the research hypothesis: nighttime incidents in higher-risk districts (especially B2, B3, and C11) show dramatically higher shooting probability. These patterns are actionable for the crime laboratory — they can be used to prioritize ballistics and DNA processing for the small subset of incidents that are most likely to involve firearms, potentially reducing backlogs while maintaining high recall of true shooting cases.

The strong temporal and geographic signals also justify the inclusion of circular time encoding and district-level features in the modeling phase. The violent offense proxy emerged as a particularly powerful early indicator.

**Next steps** (already planned in the modeling notebook):
- Merge U.S. Census ACS poverty-rate and other socioeconomic variables by district.
- Apply SMOTE to address the severe class imbalance.
- Train XGBoost with SHAP explainability.
- Build the final interactive Tableau dashboard for stakeholder delivery.

All code, plots, and data dictionaries are fully documented and reproducible in the GitHub repository.

---

## Appendix: Data Dictionary

**Boston Police Crime Incident Reports (main dataset)**

| Column                | Description                                      | Type      | Notes |
|-----------------------|--------------------------------------------------|-----------|-------|
| INCIDENT_NUMBER       | Unique incident ID                               | String    | - |
| OFFENSE_CODE_GROUP    | Broad offense category                           | String    | Used for `is_violent` proxy |
| DISTRICT              | Police district (A1, B2, etc.)                   | String    | One-hot encoded |
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
All plots and the summary table are saved in the repository. This EDA provides a solid foundation for the predictive modeling phase and directly supports forensic triage priorities for the Massachusetts State Police Crime Laboratory and Boston Police Department.