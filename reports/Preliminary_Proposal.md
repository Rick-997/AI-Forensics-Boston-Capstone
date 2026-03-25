# Preliminary Project Proposal

**Course**: DSE 6311 – Capstone  
**Submission Date**: March 2026  
**Author**: Ricardo Orellana (Solo)

## Basics
- **Who is Team Lead this week?** Ricardo Orellana  
- **Who is Recorder?** Ricardo Orellana  
- **Who is Spokesperson?** Ricardo Orellana  
- **A preliminary project title**: AI Forensic Triage Tool: Predicting Shooting Incidents in Boston to Prioritize Crime Lab Resources

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
Hypothesis: Nighttime incidents in higher-poverty districts will show ≥35% higher predicted probability of involving a shooting.  
Prediction: The XGBoost model with SHAP explainability will achieve Precision-Recall AUC > 0.80 and clearly rank poverty_rate, hour, and is_night as the top drivers.

## Data & Analysis
**What data set(s) have you found that you think are a good match for your question? Why?**  
- Boston Police Crime Incident Reports (2023–present): ~150k rows from https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system. Key columns: SHOOTING (binary target), OCCURRED_ON_DATE, DISTRICT, Lat/Long, OFFENSE_CODE_GROUP.  
- U.S. Census Bureau American Community Survey (2020–2024): Boston neighborhoods/tracts (median income, poverty rate, education, race, housing density) from https://data.census.gov.  

These datasets are public, free, large enough for robust modeling, and perfectly aligned with the research question (incident-level features + neighborhood context).

**What response / outcome variable will you use?**  
SHOOTING (binary: 1 = shooting involved, 0 = no shooting)

**What predictor variable(s) will you use?**  
Time features (hour_sin/cos, is_night, is_weekend), DISTRICT (one-hot), violent offense proxy, and district-level poverty_rate from Census ACS.

**What is your tentative analysis plan?**  
1. Data cleaning & preprocessing (parse dates, handle missing values, create binary target)  
2. Feature engineering (circular time encoding, violent proxy, district dummies, census merge, SMOTE for imbalance)  
3. Exploratory data analysis (shooting rates by district/hour, correlation matrices)  
4. Hypothesis testing (chi-square and preliminary logistic regression)  
5. Predictive modeling (XGBoost / Random Forest with SHAP explainability)

**Are there any pitfalls you can see with this plan?**  
Severe class imbalance (~0.7% shootings) and using district-level (not tract-level) poverty data. These will be addressed with SMOTE and noted as a limitation for future work.

**How will you know if your question is answered?**  
If the final model achieves PR-AUC > 0.80 and SHAP plots clearly show time and poverty as dominant predictors, the question is answered.

**How will you know if your hypothesis is supported?**  
If SHAP dependence plots and feature importance confirm that nighttime + high-poverty districts significantly increase shooting probability, the hypothesis is supported.

## Technical Details
- **What language do you plan to code in?** Python (Jupyter Notebooks)  
- **Are there any other resources you will need?** None beyond free public datasets and standard libraries (pandas, scikit-learn, XGBoost, SHAP)  
- **What is the link to your GitHub repo?** https://github.com/Rick-997/AI-Forensics-Boston-Capstone

## Planned Visualization
District heatmaps, time-of-day line plots, SHAP summary and dependence plots, and a final Tableau dashboard for stakeholder presentation.

## Expected Key Insights
Nighttime and poverty will emerge as dominant predictors. The model is expected to achieve >0.80 PR-AUC with clear, actionable SHAP explanations for forensic triage.

## Conclusion
This Boston-specific, public-data project delivers a practical AI tool with immediate public-safety value.

## References
1. Boston Police Department. (n.d.). Crime Incident Reports (August 2015 to Date). City of Boston Open Data Portal. https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system  
2. U.S. Census Bureau. (2020–2024). American Community Survey 5-Year Estimates. https://data.census.gov  
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939785  
4. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30. https://arxiv.org/abs/1705.07874  
5. Heller, S. B., et al. (2017). Thinking, fast and slow? Some field experiments to reduce crime and dropout in Chicago. The Quarterly Journal of Economics. https://doi.org/10.1093/qje/qjw033  
6. Weisburd, D., et al. (2016). Place matters: Criminology for the twenty-first century. Cambridge University Press.  
7. Braga, A. A., & Weisburd, D. (2012). The effects of focused deterrence strategies on crime: A systematic review and meta-analysis of the empirical evidence. Journal of Research in Crime and Delinquency. https://doi.org/10.1177/0022427811419368  
8. National Institute of Justice. (2021). Predictive Policing: The Role of Crime Forecasting in Law Enforcement. https://nij.ojp.gov/topics/articles/predictive-policing-role-crime-forecasting-law-enforcement