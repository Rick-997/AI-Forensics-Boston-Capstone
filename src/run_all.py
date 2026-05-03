# ========================================================
# run_all.py
# Master script to run the entire AI Forensic Triage Tool pipeline
# Works whether you run it from the repo root OR from inside the src/ folder
# ========================================================

import subprocess
from pathlib import Path

# Robust repo root detection
CURRENT_DIR = Path.cwd()
if CURRENT_DIR.name == "src":
    REPO_ROOT = CURRENT_DIR.parent          # we are inside src/
else:
    REPO_ROOT = CURRENT_DIR                 # we are in the repo root

print("🚀 Starting Full Pipeline: AI Forensic Triage Tool")
print("Capstone Project — DSE 6311\n")

scripts = [
    "01_data_wrangling.py",
    "02_eda.py",
    "03_feature_engineering_and_modeling.py",
    "04_model_evaluation_and_shap.py",
    "05_tableau_data_prep.py",
    "06_hyperparameter_tuning_and_evaluation.py",
    "07_final_model_leakage_test_and_fairness_evaluation.py",
    "08_model_comparison_leakage_mitigation_fairness.py",
]

success = True

for script in scripts:
    script_path = REPO_ROOT / "src" / script
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
        print(f"❌ ERROR: Could not find {script_path}")
        success = False
        break

if success:
    print("="*70)
    print("🎉 CONGRATULATIONS! FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("✅ All 8 scripts executed without errors")
    print("\n📁 The project is now fully up to date!")
    print("You can now open Tableau and connect to:")
    print("   visualizations/tableau/tableau_ready.csv")
else:
    print("\n⚠️  Pipeline stopped due to an error.")