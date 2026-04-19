# Baseline Models Report

**AI Forensic Triage Tool: Predicting Shooting Incidents in Boston**  
**Capstone Project — DSE 6311**  
**Author**: Ricardo Orellana  
**Date**: April 2026  

**Prepared for**: Massachusetts State Police Crime Laboratory and Boston Police Department

## Background & Question (Recap)

Forensic crime laboratories face constant pressure to process evidence quickly while working with limited staff and equipment. Ballistics analysis, DNA testing, and firearm examinations are extremely time-consuming, and every case that sits on the backlog delays justice for victims and investigators. In Boston, the police department responds to tens of thousands of incidents each year, yet only a small fraction actually involve a shooting. Still, every potential firearms-related report must be treated seriously until proven otherwise. This creates a real bottleneck: the crime lab has to decide which cases deserve immediate attention and which can safely wait.

This capstone project aims to help solve that challenge by building an AI Forensic Triage Tool. The central research question is:  
**Can incident features (time of day, location, district, offense type proxies) combined with neighborhood demographics accurately predict whether a reported crime will involve a shooting?**

We hypothesized that nighttime incidents in higher-poverty districts would show a significantly higher shooting probability — at least 35% higher than daytime or lower-poverty areas. This report covers the baseline modeling work completed so far, including the choices I made, the assumptions behind them, the results, and what I plan to do next.

## Methods

After finishing the preprocessing and feature engineering phase, I trained a baseline XGBoost classifier using the final prepared dataset (239,371 rows and 24 features). I chose XGBoost as the starting model because it performs well on tabular data, handles moderate missing values gracefully, and works naturally with SHAP values for explainability — something that will be essential when presenting results to the crime lab.

The modeling process included the following steps:
- I split the data into training and test sets using an 80/20 ratio, making sure the test set kept the original class distribution so the evaluation would be realistic.
- Because the target variable (SHOOTING) was severely imbalanced (only 0.70% positive cases), I applied SMOTE only on the training set to bring the effective shooting rate up to 50% during training.
- I trained the XGBoost model using reasonable default hyperparameters: n_estimators=200, learning_rate=0.1, and max_depth=6.
- I used Precision-Recall AUC as the primary evaluation metric because accuracy would be misleading with such a rare event.

I did not perform any hyperparameter tuning yet — this was intentionally a true baseline to establish a starting performance level and confirm that the feature engineering work was on the right track.

I also made a few explicit assumptions:
- The engineered features (night indicator, poverty rate, district encoding, etc.) capture the most important risk signals.
- The data from 2023–present is representative enough for the model to generalize to future incidents.
- Interpretability is more important than squeezing out every last bit of performance, so I avoided complex dimensionality reduction techniques like PCA.

All code for this baseline is saved in Notebook 03 and the trained model is saved in the `models/` folder.

## Results & Brief Interpretations

The baseline XGBoost model achieved a **Precision-Recall AUC of 0.8327**. For a first attempt on such an imbalanced dataset, this is a solid result and shows that the features we engineered are carrying meaningful predictive signal.

SHAP analysis provided clear and actionable insights:
- Poverty Rate stood out as the single most influential feature.
- Hour of Day and the Is Night indicator were the next strongest predictors, confirming the strong temporal pattern we expected.
- Certain districts (especially B2 and A7) also contributed noticeably to the predictions.

These findings align well with the original hypothesis. Nighttime incidents in higher-poverty areas do show significantly higher shooting risk, and the model is already picking up on those patterns. I also compared training and test performance and found the gap to be reasonable, suggesting the baseline is not severely overfitting at this stage.

To help visualize the results, I created a SHAP summary bar chart and a beeswarm plot (both saved in the visualizations folder). These plots make it easy to see which features are driving the predictions and in which direction. For the crime lab, this level of explainability is just as important as the raw performance numbers.

## Discussion & Next Steps

Overall, the baseline model confirms that the data preparation and feature engineering steps were effective. The Precision-Recall AUC of 0.8327 is encouraging, and the SHAP results match both criminological expectations and the practical knowledge shared by police officers and forensic examiners. The model is already identifying the kinds of incidents we hoped it would flag as high-risk.

That said, there is still plenty of room to improve. The next steps I plan to take are:
1. Systematic hyperparameter tuning using cross-validation to see how much performance we can gain.
2. Testing a couple of additional models (Random Forest and LightGBM) for comparison.
3. Applying stronger regularization and early stopping to further control overfitting.
4. Creating more detailed SHAP dependence plots and force plots so the crime lab can understand individual predictions.
5. Integrating the final model into the Tableau dashboard so users can explore risk scores and explanations interactively.

I will also revisit the analysis plan to make sure the next phase stays aligned with stakeholder needs. The goal is not just a high-performing model, but one that the Massachusetts State Police Crime Laboratory and Boston Police Department can actually trust and use in daily operations.

This baseline has given us a clear and promising foundation. I’m looking forward to refining the model further and delivering a practical triage tool that can help speed up forensic work and improve public safety in Boston.

---

**GitHub Repository**: https://github.com/Rick-997/AI-Forensics-Boston-Capstone  

**Live Tableau Dashboard**:  
[AI Forensic Triage Tool – Boston Shooting Risk Predictor](https://public.tableau.com/app/profile/ricardo.orellana8607/viz/AIForensicTriageTool-BostonShootingRiskPredictor/AIForensicTriageToolBostonShootingRiskMap)

All code, models, and SHAP plots are saved in the repository under the `notebooks/`, `models/`, and `visualizations/` folders.

---

