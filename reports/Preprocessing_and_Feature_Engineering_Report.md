# Pre-processing & Feature Engineering Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Prepared for**: Massachusetts State Police Crime Laboratory and Boston Police Department  

---

## Recap Background & Question

Forensic crime laboratories in Massachusetts face overwhelming caseloads. Ballistics analysis, DNA testing, and firearms examinations are extremely time-consuming and resource-intensive. Yet not every reported crime involves a shooting. Being able to quickly identify which incidents are most likely to involve a firearm would allow labs to triage evidence more effectively and deliver faster results to investigators and victims.

The central research question driving this project is:  
**Can incident features (time of day, location, district, offense type proxies) combined with neighborhood demographics accurately predict whether a reported crime will involve a shooting?**

We hypothesized that nighttime incidents in higher-poverty districts would show a significantly elevated shooting probability (at least 35% higher than daytime or lower-poverty areas). This report details the data preprocessing and feature engineering steps taken to prepare the dataset for modeling.

---

## Methods

### 1. Data Acquisition and Initial Cleaning
The primary dataset came from the Boston Police Department’s public Crime Incident Reports (August 2015–present). For this analysis we used the most recent available file (approximately 239,371 rows).  

Key preprocessing steps included:
- Removing or imputing missing values in critical fields (latitude/longitude, district, offense code group).
- Standardizing date-time fields and extracting hour, day of week, and weekend indicators.
- Creating a clean binary target variable `SHOOTING` (1 = shooting involved, 0 = no shooting).
- Filtering out clearly erroneous records (e.g., invalid coordinates outside the Boston area).

### 2. Feature Engineering
Several new features were created to capture meaningful patterns that a model could learn from:

- **Circular time encoding**: Instead of treating hour as a linear number, we created `hour_sin` and `hour_cos` to properly represent the cyclical nature of time (midnight is close to 11 PM).
- **Binary flags**: `is_night` (8 PM – 6 AM), `is_weekend`, and `is_violent` (derived from offense code group descriptions containing words like “Assault”, “Robbery”, “Shooting”, etc.).
- **Neighborhood context**: We merged district-level poverty rate from U.S. Census ACS 2020–2024 data. This gave the model important socioeconomic context without needing tract-level geocoding.
- **One-hot encoding** of the `DISTRICT` variable (19 binary columns) to allow the model to learn district-specific risk patterns.

All engineered features were chosen because they are directly interpretable by stakeholders and align with known criminological patterns (nighttime risk, socioeconomic disadvantage, violent offense history).

### 3. Handling Class Imbalance
The target variable was severely imbalanced (only 0.70% of incidents involved a shooting). We applied **SMOTE** (Synthetic Minority Over-sampling Technique) exclusively to the training set. This increased the effective shooting rate to 50% during training while leaving the test set untouched, preserving a realistic evaluation.

### 4. Unsupervised Methods Considered
Although the final model is supervised, we explored unsupervised techniques during feature exploration:
- Simple correlation analysis and heatmaps to identify redundant variables.
- We considered PCA for dimensionality reduction of the district one-hot features but ultimately decided against it because interpretability was more important for the crime lab stakeholders than a small reduction in model complexity.

These exploratory steps helped confirm that the engineered features were not highly collinear and provided meaningful signal.

### 5. Plan for Supervised Methods
The preprocessed dataset (239,371 rows × 24 features) is now ready for supervised modeling. Our plan is to:
- Train an XGBoost classifier as the primary model.
- Use Precision-Recall AUC as the main evaluation metric due to the imbalance.
- Apply SHAP values for explainability so the crime lab can understand why the model flags certain incidents.
- Perform cross-validation and hyperparameter tuning in the next phase.

All code, intermediate files, and the final engineered dataset are stored in the GitHub repository for full reproducibility.

---

## Results & Brief Interpretations

After preprocessing and feature engineering, the dataset was significantly more informative:
- The engineered time features captured clear nightly spikes in risk.
- The poverty rate variable emerged as one of the strongest signals.
- District encoding allowed the model to learn neighborhood-specific patterns (e.g., B2, B3, and C11 showed elevated risk).
- SMOTE successfully balanced the training data without introducing leakage into the test set.

Visualizations (correlation matrix, shooting rate by hour/district, and feature distributions) confirmed that the transformations aligned with domain knowledge and improved data quality for modeling.

---

## Discussion & Next Steps

The preprocessing and feature engineering phase successfully transformed raw, messy police incident data into a clean, model-ready dataset that respects both statistical best practices and stakeholder needs. Every decision — from circular time encoding to the inclusion of neighborhood poverty — was made with the goal of producing actionable, explainable predictions for the crime laboratory.

Key lessons learned:
- Domain knowledge (nighttime risk, socioeconomic factors) guided feature creation more effectively than pure automation.
- Maintaining interpretability was more valuable than aggressive dimensionality reduction.

**Next Steps (Modeling Plan)**:
1. Train and tune the XGBoost model.
2. Generate full SHAP explanations and dependence plots.
3. Build the final interactive Tableau dashboard for stakeholders.
4. Validate the model on a held-out test set and prepare a formal report for the Massachusetts State Police Crime Laboratory.

We believe the current dataset and feature set provide a strong foundation for a high-performing, trustworthy predictive tool.

---

## Appendix: Data Dictionary (Selected Key Features)

| Feature                | Type     | Description |
|------------------------|----------|-----------|
| `hour_sin` / `hour_cos`| Numeric  | Circular encoding of incident hour |
| `is_night`             | Binary   | 1 if incident occurred 8 PM–6 AM |
| `is_weekend`           | Binary   | 1 if incident on Saturday or Sunday |
| `is_violent`           | Binary   | 1 if offense group suggests violent crime |
| `poverty_rate`         | Numeric  | District-level poverty rate from ACS |
| `DISTRICT_*`           | Binary   | One-hot encoded police districts |
| `SHOOTING`             | Binary   | Target variable (1 = shooting involved) |

Full codebook and all intermediate files are available in the GitHub repository.

---

**GitHub Repository**: https://github.com/Rick-997/AI-Forensics-Boston-Capstone  

**Live Tableau Dashboard**:  
[AI Forensic Triage Tool – Boston Shooting Risk Predictor](https://public.tableau.com/app/profile/ricardo.orellana8607/viz/AIForensicTriageTool-BostonShootingRiskPredictor/AIForensicTriageToolBostonShootingRiskMap)

This report is submitted in fulfillment of M04 requirements and serves as the foundation for the final modeling and stakeholder presentation phases.