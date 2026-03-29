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
**Data set(s) chosen**  
- Boston Police Crime Incident Reports (2023–present): ~150k rows from https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system. Key columns: SHOOTING (binary target), OCCURRED_ON_DATE, DISTRICT, Lat/Long, OFFENSE_CODE_GROUP.  
- U.S. Census ACS 2020–2024: Boston neighborhoods/tracts (median income, poverty rate, education, race, housing density) from https://data.census.gov.  

These datasets are public, free, large enough for robust modeling, and perfectly aligned with the research question. I will merge the ACS data by district and plan to pull additional neighborhood variables (e.g., education attainment, housing density, median household income) to strengthen the socioeconomic signal.

**Response / outcome variable**  
SHOOTING (binary: 1 = shooting involved, 0 = no shooting)

**Predictor variable(s)**  
Time features (hour, is_night, is_weekend), DISTRICT (one-hot), violent offense proxy, and district-level poverty_rate (plus additional ACS variables) from Census ACS.

**Tentative analysis plan**  
1. Data cleaning & preprocessing (parse dates, handle missing values, create binary target)  
2. Feature engineering (circular time encoding, violent proxy, district dummies, census merge, SMOTE for imbalance)  
3. Exploratory data analysis (shooting rates by district/hour, correlation matrices)  
4. Hypothesis testing (chi-square and preliminary logistic regression)  
5. Predictive modeling (XGBoost / Random Forest with SHAP explainability)  
6. Validation: 80/20 train/test split with stratified sampling, 10-fold cross-validation, and explicit fairness checks by district and poverty level.

**Pitfalls and mitigations**  
Severe class imbalance (~0.7% shootings) will be addressed with SMOTE. Using district-level (not tract-level) poverty data is a limitation; I will note this and explore tract-level data in future work. I will also conduct a bias/fairness analysis to ensure the model does not unfairly amplify existing socioeconomic disparities.

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