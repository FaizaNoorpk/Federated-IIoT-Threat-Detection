
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "fedavg_model_70pct_10epochs_moderated.pt"
TEST_PATH = DATA_DIR / "global_test.csv"

# Match the preprocessing and network architecture in the training script.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "federated"))

from fedavg_noniid_e3 import Network

device = torch.device("cpu")

# Read the saved model's expected class order and feature columns
# from the original training script's output files where available.
metrics_path = RESULTS_DIR / "fedavg_metrics_70pct_10epochs_moderated.json"
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

test_df = pd.read_csv(TEST_PATH)
target_col = "Attack_type"

# Use the same feature columns as the saved training model.
# The feature list must match the training script exactly.
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src" / "federated"))
