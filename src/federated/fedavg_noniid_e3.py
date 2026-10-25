
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_PATH = ROOT / "models" / "fedavg_model_70pct_10epochs_moderated.pt"
TRAIN_PATH = DATA_DIR / "global_train.csv"
TEST_PATH = DATA_DIR / "global_test.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")


# ---------------------------------------------------------
# Network architecture: matches FedAvg training
# ---------------------------------------------------------
class Network(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------
# Load saved best model (no training)
# ---------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

feature_columns = checkpoint["feature_columns"]
classes = checkpoint["classes"]

print("Loaded saved model:", MODEL_PATH.name)
print("Features:", len(feature_columns))
print("Classes:", len(classes))


# ---------------------------------------------------------
# Reproduce training preprocessing
# ---------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.reindex(columns=feature_columns)
X_train = X_train.replace([np.inf, -np.inf], np.nan)

medians = X_train.median().fillna(0)
X_train = X_train.fillna(medians).fillna(0)

scaler = StandardScaler()
scaler.fit(X_train)

X_test = test_df.reindex(columns=feature_columns)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(medians).fillna(0)

X_test = scaler.transform(X_test)
X_test = np.nan_to_num(
    X_test, nan=0, posinf=0, neginf=0
)

# Match the class indices saved by LabelEncoder
class_to_index = {
    class_name: index
    for index, class_name in enumerate(classes)
}

unknown_labels = set(test_df["Attack_type"].unique()) - set(classes)
if unknown_labels:
    raise ValueError(f"Unknown test labels: {unknown_labels}")

y_true = test_df["Attack_type"].map(class_to_index).to_numpy()

X_test_tensor = torch.tensor(
    X_test, dtype=torch.float32
)


# ---------------------------------------------------------
# Predict using the saved model
# ---------------------------------------------------------
model = Network(
    input_size=len(feature_columns),
    num_classes=len(classes),
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    logits = model(X_test_tensor)
    y_pred = logits.argmax(dim=1).cpu().numpy()

print("Predictions completed. No training was performed.")


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(classes)),
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
)

csv_path = RESULTS_DIR / "fedavg_best_confusion_matrix.csv"
cm_df.to_csv(csv_path)


# ---------------------------------------------------------
# Plot confusion matrix
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 12))

image = ax.imshow(cm, interpolation="nearest", aspect="auto")
fig.colorbar(image, ax=ax, label="Number of samples")

ax.set(
    xticks=np.arange(len(classes)),
    yticks=np.arange(len(classes)),
    xticklabels=classes,
    yticklabels=classes,
    xlabel="Predicted label",
    ylabel="True label",
    title="FedAvg Confusion Matrix — Saved Best Model",
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    rotation_mode="anchor",
)

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if cm[i, j] > threshold else "black",
        )

fig.tight_layout()

figure_path = FIGURES_DIR / "fedavg_best_confusion_matrix.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\nConfusion matrix saved:", csv_path)
print("Figure saved:", figure_path)
print("\nClassification report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0,
    )
)
