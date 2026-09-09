"""Phase 0 verification — run with: python scripts/check_phase0.py"""
import sys
import pathlib

ROOT = pathlib.Path(r"c:\Users\Admin\Desktop\coding\antigrav\res projects\credit card fraud")
sys.path.insert(0, str(ROOT))

print("Testing ML package imports...")
import sklearn;    print("  scikit-learn:   ", sklearn.__version__)
import xgboost;    print("  xgboost:        ", xgboost.__version__)
import shap;       print("  shap:           ", shap.__version__)
import imblearn;   print("  imbalanced-learn:", imblearn.__version__)
import joblib;     print("  joblib:         ", joblib.__version__)
import seaborn;    print("  seaborn:        ", seaborn.__version__)
import pandas;     print("  pandas:         ", pandas.__version__)
import numpy;      print("  numpy:          ", numpy.__version__)
import flask;      print("  flask:          ", flask.__version__)
import yaml;       print("  pyyaml:         ", yaml.__version__)

print("\nTesting project modules...")
from src.utils.config import get_config, get_project_root, get_risk_thresholds
cfg = get_config()
name = cfg["project"]["name"]
version = cfg["project"]["version"]
print("  config loaded:  ", name, version)
print("  project root:   ", str(get_project_root()))

thresholds = get_risk_thresholds()
print("  risk thresholds:", thresholds)

from src.utils.logger import get_logger
logger = get_logger("test")
print("  logger:          OK")

from src.data.loader import DataLoader
loader = DataLoader()
print("  data loader:     initialized")

print("\n[OK] Phase 0 COMPLETE - All imports and modules verified.")
print("  Next step: Place creditcard.csv in data/raw/ then run:")
print("  python scripts/verify_dataset.py")
