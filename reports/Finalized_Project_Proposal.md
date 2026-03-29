# Finalized Project Proposal

**Course**: DSE 6311 – Capstone  
**Submission Date**: March 2026  
**Author**: Ricardo Orellana (Solo)

## Basics
- **Project title**: AI Forensic Triage Tool: Predicting Shooting Incidents in Boston to Prioritize Crime Lab Resources

## Background & Question

### 1) Background

#### 1.1) Domain Importance
Gunshot-related violence remains a persistent and devastating public safety challenge in major U.S. cities, and Boston is no exception. According to the Boston Police Department’s public crime incident reports, shootings represent a small but disproportionately high-impact subset of all reported crimes. Every shooting incident triggers an intensive forensic response: ballistics analysis, firearm tracing, DNA testing on recovered evidence, and often multiple rounds of laboratory work by the Massachusetts State Police Crime Laboratory. These processes are resource-heavy, time-consuming, and critical to building strong cases for prosecution. Accurate and timely forensic evidence can mean the difference between a solved case and an unsolved one, between justice for victims’ families and continued community trauma. In a city where community trust in law enforcement is essential, reliable forensic triage directly supports both public safety and equitable justice. The ability to prioritize the most serious incidents—those involving firearms—has therefore become a pressing operational need for both the Boston Police Department and the state crime laboratory that serves them.

#### 1.2) The Problem
Despite the clear importance of shooting-related cases, forensic laboratories face a fundamental bottleneck: volume. The Boston crime dataset alone contains over 150,000 incidents since 2023, yet only a tiny fraction (~0.7 %) involve a shooting. Current workflows treat every reported crime incident with the same level of forensic scrutiny because there is no automated, data-driven way to flag high-likelihood shooting cases at the moment of reporting. This uniform approach creates massive backlogs, delays ballistics and DNA processing for the most serious crimes, and strains already limited laboratory resources. Manual review of every incident report is simply not scalable. Expert analysts must currently rely on incomplete initial reports, officer intuition, or after-the-fact confirmation of a shooting—often hours or days later. In high-stakes forensic environments, this delay can compromise evidence integrity, slow investigations, and reduce the overall effectiveness of the criminal justice system. The problem is compounded by the fact that shooting incidents are not randomly distributed: they cluster by time of day (nighttime spikes), geographic district, and neighborhood socioeconomic factors. Without a predictive tool that leverages these patterns, laboratories continue to operate reactively rather than proactively.

#### 1.3) Why ML Matters (Need/Niche)
Machine learning offers a powerful, data-driven solution to this triage challenge. By training a classifier on historical incident features (time, location, district, offense type proxies) and publicly available neighborhood demographics from the U.S. Census ACS, we can generate a real-time probability score that a reported crime will involve a shooting. This probability can be delivered to the crime lab the moment an incident is logged, allowing analysts to immediately prioritize ballistics and DNA workflows for the highest-risk cases. The niche this project fills is unique: it is not broad predictive policing (which has faced criticism for bias), but rather a narrow, forensic-focused triage tool designed explicitly for laboratory resource allocation. It uses only open public data, incorporates SHAP explainability so lab analysts can understand and trust the predictions, and directly addresses the operational pain points of the Massachusetts State Police Crime Laboratory and Boston Police Department. In an era of shrinking public budgets and rising gun violence, an AI Forensic Triage Tool represents an ethical, transparent, and immediately deployable innovation that can save lives by accelerating justice in the most serious cases.

#### 1.4) Novelty
While predictive models for crime hotspots exist in the literature, the specific application of machine learning to forensic laboratory triage for shooting incidents in Boston is novel. Most prior work focuses on general crime prediction or hotspot mapping (e.g., Weisburd et al., 2016; Braga & Weisburd, 2012). Our project shifts the focus from policing to post-incident forensic prioritization, integrating real-time incident features with neighborhood-level socioeconomic data and delivering interpretable SHAP explanations tailored for forensic analysts. By combining Boston’s rich open crime data with U.S. Census ACS demographics and emphasizing explainability, this work extends existing predictive policing research into a new, high-impact domain: evidence triage. This focus on forensic lab efficiency and courtroom-ready interpretability distinguishes the project and makes it directly actionable for stakeholders who have not previously had access to such a tool.

### 2) Question

#### 2.1) Primary
Can incident features (time of day, location, district, offense type proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?

#### 2.2) Comparative
How does predictive performance vary across different machine learning approaches (e.g., XGBoost vs. Random Forest vs. logistic regression) when applied to shooting-incident classification?

#### 2.3) Interpretability
Which features—particularly time-of-day variables, district indicators, and neighborhood poverty rates—contribute most strongly to model predictions, and how consistent are feature importance patterns across models?

#### 2.4) Stakeholder Utility
Under which conditions does the model assign the highest shooting probabilities, and what does this reveal about the limitations and practical value of automated forensic triage in real-world Boston crime lab operations?

### 3) Hypotheses and Predictions

#### 3.1) Hypotheses
First, we hypothesize that nighttime incidents in higher-poverty districts will show significantly higher predicted probability of involving a shooting.  
Second, we hypothesize that non-linear ensemble methods (XGBoost) will outperform simpler linear models because the relationships between time, location, and socioeconomic factors are complex and interactive.  
Third, we hypothesize that district-level poverty rate (merged from Census ACS) will emerge as one of the top predictors, reflecting well-documented correlations between concentrated disadvantage and gun violence.

#### 3.2) Predictions
Based on these hypotheses, we expect the final XGBoost model (with SMOTE oversampling and SHAP explainability) to achieve strong predictive performance on the imbalanced shooting target. We predict that SHAP summary plots will clearly rank “is_night,” “poverty_rate,” and specific districts (e.g., B2, B3, C11) as the most influential features. We also predict that dependence plots will show a clear interaction effect: nighttime incidents in high-poverty areas will receive the highest shooting probabilities. These insights will allow the crime laboratory to operationalize the model outputs immediately, prioritizing forensic resources on the subset of incidents most likely to involve firearms.

## Data & Methods

### 1) Data

#### 1.1) Datasource #1 – Boston Police Crime Incident Reports
Our primary datasource is the Boston Police Department’s official Crime Incident Reports (August 2015–present, with a focus on the most recent full years 2023–present for model training).  
**Direct link**: https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system  

This dataset is updated daily and contains approximately 150,000–300,000 rows depending on the exact download window. Each row represents a single reported incident with rich temporal, geographic, and categorical information. The data is publicly available in CSV format, making it ideal for reproducible research. Key strengths include its official source (no sampling bias), real-time relevance to Boston’s actual crime environment, and the presence of the binary `SHOOTING` flag that serves as our ground-truth target. Limitations include occasional missing values in location fields and the fact that not every shooting is immediately flagged at the time of reporting; however, the dataset has been cleaned and standardized by the city, and we apply additional validation steps in preprocessing.

#### 1.2) Datasource #2 – U.S. Census American Community Survey (ACS) 2020–2024
To enrich the model with neighborhood-level socioeconomic context, we merge district-level and tract-level data from the U.S. Census Bureau’s American Community Survey (ACS) 5-year estimates (2020–2024).  
**Direct link**: https://data.census.gov (Boston tracts and districts via geographic filters)  

We specifically pull variables related to poverty rate, median household income, educational attainment, housing density, racial/ethnic composition, and unemployment rate. These are aggregated at the Boston Police District level for the main model (to match the granularity of the crime data) and will be explored at the tract level in sensitivity analyses. The ACS data is authoritative, annually updated, and freely available in tabular format. Merging it with the Boston crime data allows the model to capture well-documented environmental and structural factors associated with gun violence, moving beyond purely incident-level features.

#### 1.1.1) Data Dictionaries: Target & Predictor Variables

**Boston Police Crime Incident Reports (primary tabular data)**

| Column                  | Description                                      | Type          | Notes / Preprocessing |
|-------------------------|--------------------------------------------------|---------------|-----------------------|
| INCIDENT_NUMBER        | Unique incident identifier                       | String        | - |
| OFFENSE_CODE           | Numeric offense code                             | Integer       | - |
| OFFENSE_CODE_GROUP     | Broad offense category (e.g., “Aggravated Assault”) | String     | Used to create `is_violent` proxy |
| SHOOTING (target)      | Binary indicator (1 = shooting involved)         | Integer (0/1) | Severe imbalance (~0.7 %) |
| OCCURRED_ON_DATE       | Timestamp of incident                            | Datetime      | Parsed for hour, weekday, is_night |
| DISTRICT               | Police district (A1, B2, B3, etc.)               | String        | One-hot encoded |
| Lat / Long             | Geographic coordinates                           | Float         | Used for validation only |
| Location               | Street address (redacted in some releases)       | String        | - |

**U.S. Census ACS (merged by district/tract)**

| Variable                        | Description                                      | Type    | Notes |
|---------------------------------|--------------------------------------------------|---------|-------|
| poverty_rate                   | % of population below poverty line               | Float   | Primary socioeconomic predictor |
| median_household_income        | Median household income (USD)                    | Float   | - |
| education_bachelors_or_higher  | % with bachelor’s degree or higher               | Float   | - |
| housing_density                | Housing units per square mile                    | Float   | Proxy for urban density |
| unemployment_rate              | % unemployed                                     | Float   | - |
| racial_composition             | % Black, % Hispanic, % White, etc.               | Float   | Included for fairness checks only |

These dictionaries cover every variable used in the final model. Additional derived features (hour_sin, hour_cos, is_night, is_weekend, is_violent) are created during preprocessing and fully documented in the code.

### 2) Methods

#### 2.1) Data Preprocessing
All preprocessing is performed in `notebooks/01_data_wrangling.ipynb` using Python and pandas. Steps include:
- Loading the raw CSV from the public Boston data portal (no local hard-coding).
- Parsing `OCCURRED_ON_DATE` into datetime components and engineering circular time features (`hour_sin`, `hour_cos`) to capture daily periodicity.
- Creating binary flags: `is_night` (8 pm–6 am), `is_weekend`, and `is_violent` (via string matching on `OFFENSE_CODE_GROUP` with `.fillna("")` to handle any missing values).
- One-hot encoding `DISTRICT`.
- Merging ACS variables by district using a lookup dictionary (extendable to tract-level with geopandas in future iterations).
- Handling missing values (median imputation for numeric ACS variables, mode for categorical).
- Saving both a Parquet file for speed and a CSV for GitHub visibility in `data/processed/`.
- Addressing class imbalance with SMOTE oversampling inside the modeling notebook only (never on the test set).

#### Analysis Plan

##### 1) Models
We evaluate a progression of models of increasing complexity:
- Baseline: Logistic Regression (interpretable linear benchmark).
- Main model: XGBoost Classifier (`n_estimators=200`, `learning_rate=0.1`, `max_depth=6`, `eval_metric='aucpr'`).
- Alternative: Random Forest (for comparison of tree-based methods).
- Hyperparameter tuning via grid search with 10-fold cross-validation.
- All models trained after SMOTE to balance the shooting class.

##### 2) Evaluation
Performance is assessed on the imbalanced test set using:
- Precision-Recall AUC (primary metric given rarity of shootings).
- F1-score, precision, recall, and confusion matrices.
- 80/20 stratified train/test split + 10-fold cross-validation.
- Cross-dataset validation by training on 2023 data and testing on 2024–2025 hold-out periods.
- Fairness checks: stratified metrics by district and poverty quartile.

##### 3) Interpretability
SHAP (TreeExplainer) is used throughout:
- Summary bar and beeswarm plots to rank global feature importance.
- Dependence plots (e.g., `poverty_rate` interacted with `is_night`) to reveal interactions.
- Force plots for individual incident explanations that could be shown to lab analysts.

##### 4) Misclassification Analysis
We will examine false positives and false negatives by district, time of day, and poverty level to understand model limitations and identify edge cases that still require human expert review.

##### 5) What defines success (how do we know we answered the question)
Success is achieved if:
- PR-AUC exceeds 0.80 on the test set.
- SHAP plots consistently highlight expected features (nighttime, poverty, specific districts).
- The model generalizes across time periods and districts.
- Stakeholders (crime lab) receive a Tableau dashboard that turns predictions into actionable triage priorities.
- The GitHub repo and notebooks are fully reproducible and well-documented.

This comprehensive plan ensures the project is not only technically sound but directly usable by the Massachusetts State Police Crime Laboratory and Boston Police Department.
## Technical Details
- **Language**: Python (Jupyter Notebooks)  
- **Other resources needed**: None beyond free public datasets and standard libraries (pandas, scikit-learn, XGBoost, SHAP)  
- **GitHub repo**: https://github.com/Rick-997/AI-Forensics-Boston-Capstone

## Planned Visualization
District heatmaps, time-of-day line plots, SHAP summary and dependence plots, and a final Tableau dashboard for stakeholder presentation. The Tableau dashboard will include an interactive map of Boston highlighting high-risk districts by shooting probability, overlaid with poverty rates and nighttime incident density.

## Expected Key Insights
Nighttime and poverty will emerge as dominant predictors. The model is expected to provide clear, actionable SHAP explanations for forensic triage. We anticipate that districts like B3, B2, and C11 will show the highest risk during nighttime hours, allowing labs to prioritize resources effectively.

## Conclusion
This Boston-specific, public-data project delivers a practical AI tool with immediate public-safety value. Future work can expand to tract-level Census data and real-time deployment.

## References
1. Boston Police Department. Crime Incident Reports. https://data.boston.gov  
2. U.S. Census Bureau. American Community Survey. https://data.census.gov  
3. Chen, T., & Guestrin, C. (2016). XGBoost. KDD Conference.  
4. Lundberg, S. M., & Lee, S.-I. (2017). SHAP. NeurIPS.  
5. Heller, S. B., et al. (2017). Crime Reduction in Chicago. Quarterly Journal of Economics.  
6. Weisburd, D., et al. (2016). Place Matters. Cambridge University Press.  
7. Braga, A. A., & Weisburd, D. (2012). Focused Deterrence. Journal of Research in Crime and Delinquency.  
8. National Institute of Justice. (2021). Predictive Policing. https://nij.ojp.gov/topics/articles/predictive-policing-role-crime-forecasting-law-enforcement