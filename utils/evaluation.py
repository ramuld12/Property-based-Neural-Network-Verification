"""Model evaluation utilities."""

from pyexpat import features

import numpy as np
import pandas as pd
import joblib
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.preprocessing import PROPERTY_BOOLEAN_FEATURES
import os


def evaluate_model(y_true, y_pred, model_name: str = "Model", path_to_save: str = ".") -> dict:

    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    all_labels = np.unique(np.concatenate([y_true, y_pred]))

    print(f"\n=== {model_name} Classification Report ===\n")
    report = classification_report(
        y_true, y_pred, labels=all_labels, digits=4, zero_division=0, output_dict=True
    )
    report_str = classification_report(
        y_true, y_pred, labels=all_labels, digits=4, zero_division=0
    )

    print(report_str)
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print(f"\n=== Per-Label Accuracy ===\n")
    for label in all_labels:
        mask = y_true == label
        if mask.sum() > 0:
            label_accuracy = (y_pred[mask] == label).sum() / mask.sum()
            print(f"{label}: {label_accuracy:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Create one figure with two rows:
    # top = confusion matrix, bottom = classification report text
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [3, 2]}
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=all_labels,
        yticklabels=all_labels,
        ax=ax1,
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title(f"{model_name} Confusion Matrix (counts)")

    ax2.axis("off")
    ax2.text(
        0.01,
        0.99,
        f"{model_name} Classification Report\n\n{report_str}",
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    os.makedirs(path_to_save, exist_ok=True)

    plt.tight_layout()
    fig.savefig(f"{path_to_save}/{model_name}_evaluation.png", dpi=300, bbox_inches="tight")
    plt.show()

    return report


def load_and_evaluate_model(
    joblib_path: str,
    X: pd.DataFrame,
    y_true,
    model_name: str = "Model",
    device=None,
    batch_size: int = 1024,
    path_to_save: str = ".",
):
    joblib_object = joblib.load(joblib_path)

    model = joblib_object["model"]
    ordinal_encoder = joblib_object.get("ordinal_encoder")
    scaler = joblib_object.get("scaler")
    label_encoder = joblib_object.get("label_encoder")

    features = joblib_object["features"]
    categorical_cols = joblib_object.get("categorical_cols", [])
    continuous_cols = joblib_object.get("continuous_cols")
    binary_cols = joblib_object.get("binary_cols")

    if binary_cols is None:
        binary_cols = [c for c in PROPERTY_BOOLEAN_FEATURES if c in features]

    if continuous_cols is None:
        continuous_cols = [c for c in features if c not in binary_cols]

    X = X.copy()
    X = X[features].copy()

    if categorical_cols:
        X[categorical_cols] = ordinal_encoder.transform(X[categorical_cols])

    X[continuous_cols] = X[continuous_cols].apply(pd.to_numeric, errors="coerce")
    X[continuous_cols] = X[continuous_cols].replace([np.inf, -np.inf], np.nan)
    X[continuous_cols] = X[continuous_cols].fillna(0.0)

    X[binary_cols] = X[binary_cols].apply(pd.to_numeric, errors="coerce")
    X[binary_cols] = X[binary_cols].fillna(0).astype(int)

    X_scaled_df = X.copy()

    if scaler is not None and len(continuous_cols) > 0:
        X_scaled_df[continuous_cols] = scaler.transform(X[continuous_cols])

    for col in binary_cols:
        X_scaled_df[col] = X[col].values

    X_np = X_scaled_df[features].values.astype(np.float32)

    # ==================================================
    # Random Forest
    # ==================================================
    if hasattr(model, "predict") and not isinstance(model, torch.nn.Module):
        y_pred = model.predict(X_np)

    # ==================================================
    # PyTorch model
    # ==================================================
    else:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)

        preds_all = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                xb = X_tensor[i:i + batch_size].to(device)

                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                preds_all.extend(preds)

        y_pred = label_encoder.inverse_transform(np.array(preds_all))

    return evaluate_model(
        y_true,
        y_pred,
        model_name=model_name,
        path_to_save=path_to_save,
    )
