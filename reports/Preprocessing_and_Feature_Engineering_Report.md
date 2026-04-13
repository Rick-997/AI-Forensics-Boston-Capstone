# Pre-processing & Feature Engineering Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Prepared for**: Massachusetts State Police Crime Laboratory and Boston Police Department

## Recap Background & Question

Forensic crime laboratories across the United States, including the Massachusetts State Police Crime Laboratory, operate under intense pressure. The volume of cases continues to grow while staffing and equipment resources remain limited. Ballistics analysis, DNA testing, firearm examinations, and other specialized forensic work are extremely time-consuming processes that can take weeks or even months to complete. Every hour spent on a low-priority case delays justice for victims of serious violent crimes.

In Boston, the situation is particularly challenging. The Boston Police Department responds to tens of thousands of reported incidents each year, yet only a small fraction of those incidents actually involve a shooting. Despite this, every reported crime with a potential firearm component must be treated as a possible shooting until proven otherwise. This creates a significant bottleneck: the crime lab must decide which cases deserve immediate forensic attention and which can safely wait. Without an objective, data-driven way to prioritize, valuable laboratory resources are often spread too thin, slowing investigations and reducing the overall effectiveness of the criminal justice system.

This capstone project directly addresses that real-world challenge by developing an AI Forensic Triage Tool. The goal is to help forensic analysts and police leadership quickly identify which reported incidents are most likely to involve a shooting so that limited laboratory resources — such as ballistics testing, shell casing analysis, and DNA processing — can be focused on the cases with the highest public safety impact.

The central research question guiding the entire project is:

Can incident features (time of day, location, district, offense type proxies) combined with neighborhood demographics accurately predict whether a reported crime will involve a shooting?

We hypothesized that certain patterns would emerge strongly from the data. Specifically, we predicted that nighttime incidents occurring in higher-poverty districts would show a significantly elevated probability of involving a shooting — at least 35% higher compared to daytime incidents or those in lower-poverty areas. This hypothesis is grounded in both criminological research and the lived experience of Boston police officers and forensic examiners who have long observed that violence tends to concentrate in specific times and neighborhoods.

By building a predictive model that can assign a reliable shooting probability score to each new incident, the project aims to give the Massachusetts State Police Crime Laboratory and the Boston Police Department a practical, explainable tool they can use in daily operations. Ultimately, the hope is that faster triage will lead to quicker forensic results, stronger cases, and faster justice for victims and communities across Boston.

## Methods

The preprocessing and feature engineering phase was designed with two main goals in mind: (1) prepare a clean, high-quality dataset that a supervised model could learn from effectively, and (2) create features that would be both statistically powerful and easily understandable by the Massachusetts State Police Crime Laboratory and Boston Police Department stakeholders. Every decision was made with interpretability and real-world usability in mind rather than maximizing model complexity.

### 1. Data Acquisition and Initial Cleaning
The project began with the publicly available Boston Police Crime Incident Reports dataset (2023–present), which contained approximately 239,371 rows at the start of analysis. The raw data was loaded using the `pandas` library from a locally stored parquet file for efficiency.

Initial cleaning steps included:
- Removing or appropriately imputing missing values in critical columns such as latitude, longitude, district, and offense code group.
- Standardizing date-time fields and extracting the hour of the incident.
- Filtering out clearly erroneous records (e.g., coordinates falling well outside the Boston metropolitan area).
- Creating a clean binary target variable `SHOOTING` (1 = shooting involved, 0 = no shooting).

All cleaning operations were documented in Notebook 01 (`01_data_wrangling.ipynb`) with detailed comments so the process remains fully reproducible.

### 2. Feature Engineering
After basic cleaning, we engineered several new features that capture meaningful behavioral and contextual patterns:

- **Circular time encoding**: Instead of treating the hour of the day as a simple linear number (0–23), we created `hour_sin` and `hour_cos`. This prevents the model from treating 11 PM and 1 AM as far apart when they are actually close in the daily cycle.
- **Binary indicator variables**:
  - `is_night`: 1 if the incident occurred between 8 PM and 6 AM (a period historically associated with higher violence).
  - `is_weekend`: 1 if the incident occurred on Saturday or Sunday.
  - `is_violent`: 1 if the offense code group description contained keywords suggesting violent crime (e.g., Assault, Robbery, Shooting). This proxy was created using string matching on the `OFFENSE_CODE_GROUP` column.
- **Neighborhood socioeconomic context**: We merged district-level poverty rate data from the U.S. Census American Community Survey (ACS) 2020–2024. This added a single numeric column `poverty_rate` that gives the model important contextual information about the economic conditions of the neighborhood where the incident occurred.
- **Categorical encoding**: The `DISTRICT` variable was one-hot encoded into 19 binary columns so the model could learn district-specific risk patterns without assuming any artificial ordering.

These features were chosen deliberately because they are grounded in both criminological literature and practical knowledge shared by police officers and forensic examiners. Each engineered feature is directly interpretable, which is essential for stakeholder trust.

### 3. Unsupervised Methods Considered
Although the final goal is a supervised prediction model, we performed several unsupervised exploratory steps to better understand the data structure before moving to modeling:

- Correlation analysis and heatmap visualizations were generated in Notebook 02 (`02_eda.ipynb`) to identify potential multicollinearity and confirm that the engineered features provided independent signals.
- We briefly considered Principal Component Analysis (PCA) as a dimensionality reduction technique for the one-hot encoded district variables. However, after reviewing the results, we decided against using PCA because reducing interpretability would make it much harder for the crime lab to understand and trust the model’s predictions. In a real-world forensic setting, explainability is more valuable than a small gain in computational efficiency.

These unsupervised explorations helped validate our feature choices and guided the final supervised modeling plan.

### 4. Handling Class Imbalance
The target variable `SHOOTING` was severely imbalanced (only 0.70% of incidents involved a shooting). To address this, we applied **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively to the training set. This increased the effective shooting rate in the training data to 50% while leaving the test set completely untouched, ensuring a realistic and unbiased evaluation.

### 5. Plan for Supervised Methods
With the cleaned and engineered dataset now ready (239,371 rows × 24 features), we are well positioned to move into supervised modeling. The plan includes:

- Training an **XGBoost classifier** as the primary model due to its strong performance on tabular data, built-in handling of missing values, and excellent interpretability when combined with SHAP.
- Using **Precision-Recall AUC** as the main evaluation metric because of the severe class imbalance.
- Generating **SHAP values** for every prediction to provide clear, instance-level explanations that the crime lab can use to understand why a particular incident received a high (or low) shooting probability score.
- Performing cross-validation and hyperparameter tuning in the next phase to ensure the model generalizes well.

All preprocessing code, engineered features, and the final prepared dataset are saved in the GitHub repository under `data/processed/` and `notebooks/`. This ensures full reproducibility and allows the stakeholder to trace every step from raw data to final model input.

---

## Results & Brief Interpretations

After completing the full preprocessing and feature engineering pipeline, the dataset was transformed from raw, messy police incident records into a clean, model-ready feature matrix. The final dataset contained 239,371 rows and 24 engineered features. This represents a substantial improvement in data quality and usability compared to the original raw file.

### Key Outcomes from Preprocessing
- Missing values in critical columns (latitude/longitude, district, offense code group) were handled appropriately without introducing bias.
- All date-time fields were standardized, allowing reliable extraction of temporal patterns.
- The binary target variable `SHOOTING` was created cleanly, confirming the severe class imbalance: only **0.70%** of incidents involved a shooting.

### Feature Engineering Results
The engineered features provided rich, interpretable signals:
- Circular time encoding (`hour_sin` and `hour_cos`) successfully captured the cyclical nature of the day, preventing the model from treating midnight and 11 PM as distant.
- Binary flags (`is_night`, `is_weekend`, `is_violent`) highlighted well-known risk periods and offense types.
- The merged district-level `poverty_rate` from U.S. Census ACS data added important neighborhood context that would otherwise have been missing.
- One-hot encoding of the 19 police districts allowed the model to learn location-specific risk patterns.

### Class Imbalance Handling
Because the target was so imbalanced, SMOTE was applied only to the training set. This increased the effective shooting rate in the training data to 50%, while the test set remained untouched at the original 0.70% rate. This approach ensured the model learned from a balanced training environment without contaminating the evaluation.

### Exploratory Analysis Highlights
Unsupervised exploration in Notebook 02 revealed several important patterns:
- Nighttime incidents showed a dramatically higher shooting rate (1.59%) compared to daytime (0.31%) — a roughly 5× increase.
- Districts B2, B3, and C11 consistently showed the highest shooting probabilities, aligning with known higher-poverty neighborhoods.
- The correlation heatmap confirmed that the engineered features were not highly collinear, supporting their independent value.

### Supervised Modeling Readiness
The final feature matrix was saved as both a parquet file and a CSV for easy use in modeling. An initial XGBoost baseline model trained on this data achieved a **Precision-Recall AUC of 0.8327**, which is a strong result given the extreme class imbalance.

SHAP analysis on the trained model provided clear interpretability:
- **Poverty Rate** emerged as the single most influential feature.
- **Hour of Day** and **Is Night** were the next strongest predictors, confirming the strong temporal component of risk.
- Specific districts (especially B2 and A7) also contributed meaningfully.

These results strongly support the original hypothesis: nighttime incidents in higher-poverty districts do indeed carry significantly higher shooting risk. The model is now ready for the next phase of hyperparameter tuning, cross-validation, and full stakeholder-facing explainability work.

All results, including the engineered dataset, SHAP plots, and performance metrics, are saved in the GitHub repository under the `models/` and `visualizations/` folders for full transparency and reproducibility.
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