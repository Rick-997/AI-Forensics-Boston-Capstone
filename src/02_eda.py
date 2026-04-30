# Exploratory Data Analysis (EDA)


# ================== PATHS & IMPORTS ==================
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Robust path that works whether you run from notebooks/ or root
CURRENT_DIR = Path.cwd()
REPO_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
VISUALIZATIONS = REPO_ROOT / "visualizations"
VISUALIZATIONS.mkdir(parents=True, exist_ok=True)

print(f"✅ Loading from: {DATA_PROCESSED}")
print(f"✅ Plots will be saved to: {VISUALIZATIONS}")


# Load the cleaned parquet (fastest format)
df = pd.read_parquet(DATA_PROCESSED / "crimes_cleaned.parquet")

print(f"Dataset shape: {df.shape}")
print(f"Shooting rate: {df['SHOOTING'].mean():.4%}")
df.head()

# 1. Overall Shooting Rate & Class Imbalance

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='SHOOTING', palette=['#1f77b4', '#ff7f0e'])
plt.title('Shooting vs Non-Shooting Incidents (Severe Imbalance)')
plt.ylabel('Number of Incidents')
plt.xlabel('Shooting (0 = No, 1 = Yes)')
plt.text(0.5, df['SHOOTING'].value_counts()[0]*0.95, 
         f'{df["SHOOTING"].value_counts()[0]:,} incidents', ha='center')
plt.text(1.5, df['SHOOTING'].value_counts()[1]*0.95, 
         f'{df["SHOOTING"].value_counts()[1]:,} incidents', ha='center')
plt.savefig(VISUALIZATIONS / "01_shooting_imbalance.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Temporal Analysis (Time of Day & Night vs Day)

hourly = df.groupby('hour')['SHOOTING'].mean().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=hourly, x='hour', y='SHOOTING', marker='o', linewidth=2.5)
plt.title('Shooting Probability by Hour of Day')
plt.ylabel('Proportion of Incidents Involving Shooting')
plt.xlabel('Hour (0-23)')
plt.axvspan(18, 24, alpha=0.2, color='red', label='Night (6PM-6AM)')
plt.axvspan(0, 6, alpha=0.2, color='red')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(VISUALIZATIONS / "02_shooting_by_hour.png", dpi=300, bbox_inches='tight')
plt.show()

night_stats = df.groupby('is_night')['SHOOTING'].mean()

plt.figure(figsize=(8,5))
sns.barplot(x=night_stats.index, y=night_stats.values, palette=['#1f77b4', '#ff7f0e'])
plt.title('Shooting Rate: Night vs Day')
plt.ylabel('Shooting Probability')
plt.xlabel('Is Night (6PM–6AM)')
for i, v in enumerate(night_stats.values):
    plt.text(i, v + 0.001, f'{v:.2%}', ha='center')
plt.savefig(VISUALIZATIONS / "03_night_vs_day.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. District Analysis (Highest Risk Areas)

district_rates = df.groupby('DISTRICT')['SHOOTING'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=district_rates.index, y=district_rates.values, palette='viridis')
plt.title('Shooting Probability by Police District')
plt.ylabel('Shooting Rate')
plt.xlabel('District')
plt.xticks(rotation=45)
for i, v in enumerate(district_rates.values):
    plt.text(i, v + 0.001, f'{v:.2%}', ha='center')
plt.savefig(VISUALIZATIONS / "04_district_rates.png", dpi=300, bbox_inches='tight')
plt.show()

print("Top 3 highest-risk districts:\n", district_rates.head(3))

# 4. Hour × District Heatmap (Key Hypothesis Visual)

heatmap_data = df.pivot_table(index='hour', columns='DISTRICT', values='SHOOTING', aggfunc='mean')

plt.figure(figsize=(14,8))
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.1%', linewidths=0.5)
plt.title('Shooting Probability Heatmap: Hour of Day × District')
plt.ylabel('Hour of Day')
plt.xlabel('District')
plt.savefig(VISUALIZATIONS / "05_hour_district_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. Violent Offense Proxy & Correlations

# ================== FIXED: Violent Offense Proxy & Correlations ==================
# Safety check – recreate 'is_violent' if missing from old parquet
if 'is_violent' not in df.columns:
    print("⚠️ 'is_violent' column missing from parquet – recreating it now...")
    df['is_violent'] = (df['OFFENSE_CODE_GROUP']
                        .fillna('')
                        .astype(str)
                        .str.contains("Assault|Robbery|Homicide|Murder", 
                                      case=False, na=False)
                        .astype(int))

violent_rate = df.groupby('is_violent')['SHOOTING'].mean()

print(f"✅ is_violent unique values: {df['is_violent'].unique()}")
print("Shooting rate when violent offense :", violent_rate.get(1, 0))
print("Shooting rate when non-violent    :", violent_rate.get(0, 0))

corr = df[['SHOOTING', 'hour', 'is_night', 'is_weekend', 'is_violent']].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix (Key Features)')
plt.savefig(VISUALIZATIONS / "06_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# 6. Key Insights & Hypothesis Validation

summary = pd.DataFrame({
    'Metric': ['Overall Shooting Rate', 'Night Shooting Rate', 'Day Shooting Rate', 'Highest District Rate'],
    'Value': [df['SHOOTING'].mean(), 
              df[df['is_night']==1]['SHOOTING'].mean(),
              df[df['is_night']==0]['SHOOTING'].mean(),
              district_rates.max()]
})
summary.to_csv(VISUALIZATIONS / "eda_summary_table.csv", index=False)
print("✅ All plots and summary saved to visualizations/ folder!")
