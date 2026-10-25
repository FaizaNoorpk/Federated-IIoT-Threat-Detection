import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. Model comparison
# ---------------------------------------------------------
models = [
    "Centralized NN",
    "IID FedAvg",
    "Non-IID FedAvg",
    "Extra Trees",
]

accuracy = [
    0.7437,
    0.7615827383,
    0.7323624697,
    0.9096341962,
]

macro_f1 = [
    0.7099,
    0.7330904116,
    0.6743013330,
    0.8729978250,
]

weighted_f1 = [
    0.7497,
    0.7672093977,
    0.7160545584,
    0.9119701450,
]

x = range(len(models))

plt.figure(figsize=(10, 6))

plt.bar(
    [i - 0.25 for i in x],
    accuracy,
    width=0.25,
    label="Accuracy",
)

plt.bar(
    x,
    macro_f1,
    width=0.25,
    label="Macro-F1",
)

plt.bar(
    [i + 0.25 for i in x],
    weighted_f1,
    width=0.25,
    label="Weighted-F1",
)

plt.xticks(list(x), models)
plt.ylabel("Score")
plt.xlabel("Model")
plt.title("Performance Comparison on Held-Out Test Set")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "model_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# 2. Extra Trees per-class F1
# ---------------------------------------------------------
# ---------------------------------------------------------
# 2. Extra Trees per-class F1
# ---------------------------------------------------------
report_path = RESULTS_DIR / "extra_trees_test_report.csv"

if report_path.exists():
    report = pd.read_csv(report_path, index_col=0)

    if "f1-score" in report.columns:
        excluded_rows = ["accuracy", "macro avg", "weighted avg"]

        class_rows = report.loc[
            ~report.index.astype(str).isin(excluded_rows)
        ]

        plt.figure(figsize=(12, 7))
        plt.bar(
            class_rows.index.astype(str),
            class_rows["f1-score"],
        )

        plt.ylabel("F1-score")
        plt.xlabel("Attack Class")
        plt.title("Extra Trees F1-score by Attack Class")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "extra_trees_per_class_f1.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Generated: extra_trees_per_class_f1.png")
    else:
        print("Could not find the 'f1-score' column:", report.columns.tolist())
else:
    print("Report file not found:", report_path)


# ---------------------------------------------------------
# 3. Extra Trees confusion matrix
# ---------------------------------------------------------
cm_path = RESULTS_DIR / "extra_trees_test_confusion_matrix.csv"

if cm_path.exists():

    cm = pd.read_csv(cm_path, index_col=0)

    plt.figure(figsize=(11, 9))

    plt.imshow(cm.values)

    plt.colorbar(label="Number of Samples")

    plt.xticks(
        range(len(cm.columns)),
        cm.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(cm.index)),
        cm.index,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Extra Trees Confusion Matrix")

    # Add numbers inside cells
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            plt.text(
                j,
                i,
                str(cm.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "extra_trees_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------
# 4. IID FedAvg training history
# ---------------------------------------------------------
iid_history_path = RESULTS_DIR / "fedavg_iid_e3_history.json"

if iid_history_path.exists():
    with open(iid_history_path, "r") as f:
        history = json.load(f)

    if history and "train_loss" in history[0]:
        rounds = [item["round"] for item in history]
        losses = [item["train_loss"] for item in history]
        accuracies = [item["train_accuracy"] for item in history]

        # Training loss curve
        plt.figure(figsize=(9, 6))
        plt.plot(rounds, losses, marker="o")
        plt.xlabel("Communication Round")
        plt.ylabel("Training Loss")
        plt.title("IID FedAvg Training Loss")
        plt.xticks(rounds)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "fedavg_iid_training_loss.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Training accuracy curve
        plt.figure(figsize=(9, 6))
        plt.plot(rounds, accuracies, marker="o")
        plt.xlabel("Communication Round")
        plt.ylabel("Training Accuracy")
        plt.title("IID FedAvg Training Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rounds)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "fedavg_iid_training_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Generated IID FedAvg training curves.")
    else:
        print("IID history has no training metrics.")

# ---------------------------------------------------------
# 5. Non-IID FedAvg training history
# ---------------------------------------------------------
noniid_history_path = RESULTS_DIR / "fedavg_history.json"

if noniid_history_path.exists():

    with open(noniid_history_path, "r") as f:
        history = json.load(f)

    print("\nNon-IID FedAvg history structure:")
    print(history)

    if isinstance(history, dict):

        rounds = history.get("round", history.get("rounds"))
        losses = history.get("loss", history.get("losses"))

        if rounds is not None and losses is not None:

            plt.figure(figsize=(9, 6))

            plt.plot(
                rounds,
                losses,
                marker="o",
            )

            plt.xlabel("Communication Round")
            plt.ylabel("Training Loss")
            plt.title("Non-IID FedAvg Training Loss")

            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(
                FIGURES_DIR / "fedavg_noniid_training_loss.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()


print("\nFigures generated successfully.")
print(f"Output folder: {FIGURES_DIR}")