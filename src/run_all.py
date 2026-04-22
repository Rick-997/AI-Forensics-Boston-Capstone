# ========================================================
# run_all.py
# Master script to run the entire AI Forensic Triage Tool pipeline
# ========================================================

import subprocess
from pathlib import Path

print("🚀 Starting Full Pipeline: AI Forensic Triage Tool")
print("Capstone Project — DSE 6311\n")

scripts = [
    "src/01_data_wrangling.py",
    "src/02_eda.py",
    "src/03_feature_engineering_and_modeling.py",
    "src/04_model_evaluation_and_shap.py",
    "src/05_tableau_data_prep.py"
]

success = True

for script in scripts:
    script_path = Path(script)
    print(f"\n{'='*70}")
    print(f"▶️  Running: {script}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            check=True,
            capture_output=False
        )
        print(f"✅ {script} completed successfully!\n")
        
    except subprocess.CalledProcessError:
        print(f"❌ ERROR: {script} failed!")
        success = False
        break
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find {script}")
        success = False
        break

if success:
    print("="*70)
    print("🎉 CONGRATULATIONS! ALL SCRIPTS RAN SUCCESSFULLY!")
    print("="*70)
    print("✅ Data wrangling completed")
    print("✅ EDA completed")
    print("✅ Feature engineering + XGBoost model completed")
    print("✅ Model evaluation + SHAP plots completed")
    print("✅ Tableau-ready CSV generated")
    print("\n📁 Your project is now fully up to date!")
    print("You can now open Tableau and connect to:")
    print("   visualizations/tableau/tableau_ready.csv")
else:
    print("\n⚠️  Pipeline stopped due to an error.")