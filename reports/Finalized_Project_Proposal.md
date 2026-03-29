# Finalized Project Proposal

**Course**: DSE 6311 – Capstone  
**Submission Date**: March 2026  
**Author**: Ricardo Orellana (Solo)

## Basics
- **Project title**: AI Forensic Triage Tool: Predicting Shooting Incidents in Boston to Prioritize Crime Lab Resources

## Background & Question
**A defined research question that serves a need or fills a niche**  
Can incident features (time of day, location, district, offense type proxies) and neighborhood demographics accurately predict whether a reported crime will involve a shooting?

**What is the question?**  
The question asks whether readily available public data can forecast shooting involvement in real time.

**What need or niche does it fill?**  
Forensic crime laboratories and the Boston Police Department process thousands of incidents annually under tight resource constraints. Ballistics analysis, DNA testing, and firearms-related evidence are extremely time-intensive. However, not every reported crime involves a shooting. Current workflows treat all incidents uniformly, creating backlogs and delaying justice in shooting-related cases. This tool fills the niche of real-time forensic triage so high-impact cases can be prioritized.

**Why is it worth your time/effort to explore this question?**  
It directly supports the Massachusetts State Police Crime Laboratory and Boston Police Department by enabling faster evidence processing for shooting cases. The project uses only free public data, meets every capstone requirement, and delivers immediate stakeholder value.

**Is your question novel / original?**  
It is not entirely novel (predictive policing exists), but applying it specifically to forensic lab triage with SHAP explainability for a real-world stakeholder in Boston is original and actionable.

**An identified stakeholder**  
Massachusetts State Police Crime Laboratory + Boston Police Department (they will use the model outputs to prioritize ballistics/DNA processing).

**A hypothesis and prediction**  
Hypothesis: Nighttime incidents in higher-poverty districts will show higher predicted probability of involving a shooting.  
Prediction: The XGBoost model with SHAP explainability will demonstrate strong predictive performance and clearly identify time-of-day and poverty as the top drivers.

## Data & Methods
**Data set(s) chosen**  
- Boston Police Crime Incident Reports (2023–present): ~150k rows from https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system. Key columns: SHOOTING (binary target), OCCURRED_ON_DATE, DISTRICT, Lat/Long, OFFENSE_CODE_GROUP.  
- U.S. Census ACS 2020–2024: Boston neighborhoods/tracts (median income, poverty rate, education, race, housing density) from https://data.census.gov.  

These datasets are public, free, large enough for robust modeling, and perfectly aligned with the research question (incident-level features + neighborhood context). I will merge the ACS data by district and plan to pull additional neighborhood variables (e.g., education attainment, housing density) to strengthen the socioeconomic signal.

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
District heatmaps, time-of-day line plots, SHAP summary and dependence plots, and a final Tableau dashboard for stakeholder presentation.

## Expected Key Insights
Nighttime and poverty will emerge as dominant predictors. The model is expected to provide clear, actionable SHAP explanations for forensic triage.

## Conclusion
This Boston-specific, public-data project delivers a practical AI tool with immediate public-safety value.

## References
1. Boston Police Department. Crime Incident Reports. https://data.boston.gov  
2. U.S. Census Bureau. American Community Survey. https://data.census.gov  
3. Chen, T., & Guestrin, C. (2016). XGBoost. KDD Conference.  
4. Lundberg, S. M., & Lee, S.-I. (2017). SHAP. NeurIPS.  
5. Heller, S. B., et al. (2017). Crime Reduction in Chicago. Quarterly Journal of Economics.  
6. Weisburd, D., et al. (2016). Place Matters. Cambridge University Press.  
7. Braga, A. A., & Weisburd, D. (2012). Focused Deterrence. Journal of Research in Crime and Delinquency.  
8. National Institute of Justice. (2021). Predictive Policing. https://nij.ojp.gov/topics/articles/predictive-policing-role-crime-forecasting-law-enforcement