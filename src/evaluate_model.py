# src/evaluate_model.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pathlib import Path

REPORTS_DIR = Path("reports")
PLOTS_DIR = Path("plots")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def save_classification_report(y_true, y_pred, out_path: Path = REPORTS_DIR / "classification_report.csv"):
    rep = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(rep).transpose().to_csv(out_path)
    print("Saved classification report to", out_path)

def save_confusion_matrix(y_true, y_pred, out_path: Path = PLOTS_DIR / "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved confusion matrix to", out_path)

def save_roc_auc(y_true, y_proba, out_path: Path = REPORTS_DIR / "roc_auc.txt"):
    if y_proba is None:
        print("No probabilities provided; skipping ROC AUC save.")
        return
    auc = roc_auc_score(y_true, y_proba)
    with open(out_path, "w") as f:
        f.write(f"{auc:.6f}\n")
    print("Saved ROC AUC to", out_path)
    return auc

def save_feature_importances(feature_names, importances, out_path: Path = REPORTS_DIR / "feature_importances.csv"):
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(out_path, index=False)
    print("Saved feature importances to", out_path)
