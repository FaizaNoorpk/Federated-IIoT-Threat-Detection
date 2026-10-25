
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 512
LEARNING_RATE = 0.001

ROOT = Path(".")
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")


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


def load_features(path, columns, medians, scaler):
    df = pd.read_csv(path)
    X = df.reindex(columns=columns)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(medians).fillna(0)
    X = scaler.transform(X)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = label_encoder.transform(df["Attack_type"])

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def train_client(global_state, X, y, class_weights):
    model = Network(len(feature_columns), num_classes).to(device)
    model.load_state_dict(copy.deepcopy(global_state))
    model.train()

    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return copy.deepcopy(model.state_dict()), len(dataset)


def fedavg(client_states, client_sizes):
    total_size = sum(client_sizes)
    global_state = copy.deepcopy(client_states[0])

    for key in global_state:
        if torch.is_floating_point(global_state[key]):
            global_state[key] = sum(
                client_states[i][key] * (client_sizes[i] / total_size)
                for i in range(len(client_states))
            )
        else:
            global_state[key] = client_states[0][key]

    return global_state
def evaluate_model(model, X, y):
    model.eval()

    with torch.no_grad():
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, y).item()
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()

    return loss, accuracy

# Read training data to fit preprocessing on training records only.
train_df = pd.read_csv(DATA_DIR / "global_train.csv")
feature_columns = [
    col for col in train_df.columns if col != "Attack_type"
]

label_encoder = LabelEncoder()
label_encoder.fit(train_df["Attack_type"])
num_classes = len(label_encoder.classes_)

X_raw = train_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)
medians = X_raw.median().fillna(0)
X_imputed = X_raw.fillna(medians).fillna(0)

scaler = StandardScaler()
scaler.fit(X_imputed)
X_scaled = scaler.transform(X_imputed)
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

X_train_eval = torch.tensor(X_scaled, dtype=torch.float32)
y_train_eval = torch.tensor(
    label_encoder.transform(train_df["Attack_type"]),
    dtype=torch.long,
)

# Calculate class weights from the pooled training labels.
# This is a simulation convenience, not a fully decentralized setup.
y_all = label_encoder.transform(train_df["Attack_type"])
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_all,
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

client_files = [
    DATA_DIR / "client_iid_1_train.csv",
    DATA_DIR / "client_iid_2_train.csv",
    DATA_DIR / "client_iid_3_train.csv" 
]

clients = []
for path in client_files:
    X_client, y_client = load_features(
        path, feature_columns, medians, scaler
    )
    clients.append((X_client, y_client))
    print(f"Loaded {path.name}: {len(y_client):,} records")

global_model = Network(len(feature_columns), num_classes).to(device)
history = []

print("\nStarting FedAvg training...")
for round_number in range(1, ROUNDS + 1):
    client_states = []
    client_sizes = []

    for client_number, (X_client, y_client) in enumerate(clients, start=1):
        state, size = train_client(
            global_model.state_dict(),
            X_client,
            y_client,
            class_weights,
        )
        client_states.append(state)
        client_sizes.append(size)

    global_model.load_state_dict(
        fedavg(client_states, client_sizes)
    )
    train_loss, train_accuracy = evaluate_model(
        global_model,
        X_train_eval,
        y_train_eval,
    )

    record = {
        "round": round_number,
        "client_sizes": client_sizes,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
    }
    history.append(record)

    print(
        f"Round {round_number}/{ROUNDS} | "
        f"Training Loss: {train_loss:.4f} | "
        f"Training Accuracy: {train_accuracy:.4f}"
    )

# Evaluate the final global model on the untouched test set.
test_X, test_y = load_features(
    DATA_DIR / "global_test.csv",
    feature_columns,
    medians,
    scaler,
)

global_model.eval()
with torch.no_grad():
    logits = global_model(test_X)
    predictions = logits.argmax(dim=1).numpy()

y_true = test_y.numpy()

metrics = {
    "model": "FedAvg neural network",
    "clients": len(clients),
    "communication_rounds": ROUNDS,
    "local_epochs": LOCAL_EPOCHS,
    "features": len(feature_columns),
    "test_records": len(y_true),
    "accuracy": float(accuracy_score(y_true, predictions)),
    "macro_f1": float(
        f1_score(y_true, predictions, average="macro", zero_division=0)
    ),
    "weighted_f1": float(
        f1_score(y_true, predictions, average="weighted", zero_division=0)
    ),
    "client_sizes": [len(y) for _, y in clients],
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

(RESULTS_DIR / "fedavg_iid_e3_metrics.json").write_text(
    json.dumps(metrics, indent=2)
)
(RESULTS_DIR / "fedavg_iid_e3_history.json").write_text(
    json.dumps(history, indent=2)
)
pd.DataFrame(
    classification_report(
        y_true,
        predictions,
        labels=np.arange(num_classes),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
).transpose().to_csv(RESULTS_DIR / "fedavg_iid_e3_test_report.csv")

torch.save(
    {
        "model_state_dict": global_model.state_dict(),
        "feature_columns": feature_columns,
        "classes": list(label_encoder.classes_),
        "metrics": metrics,
    },
    MODELS_DIR / "fedavg_model.pt",
)

print("\nFedAvg held-out test results:")
print(json.dumps(metrics, indent=2))
print("\nPer-class report:")
print(classification_report(
    y_true,
    predictions,
    labels=np.arange(num_classes),
    target_names=label_encoder.classes_,
    zero_division=0,
))
print("\nSaved results/fedavg_iid_e3_metrics.json")
print("Saved results/fedavg_iid_e3_history.json")
print("Saved results/fedavg_iid_e3_test_report.csv")
print("Saved models/fedavg_iid_e3_model.pt")
