"""Model evaluation utilities."""

import numpy as np
import pandas as pd
import joblib
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """Evaluate model predictions and display results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of model for display purposes
        
    Returns:
        Classification report dictionary
    """
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
    print(classification_report(y_true, y_pred, labels=all_labels, digits=4))
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    # Print per-label accuracy
    print(f"\n=== Per-Label Accuracy ===\n")
    for label in all_labels:
        mask = y_true == label
        if mask.sum() > 0:
            label_accuracy = (y_pred[mask] == label).sum() / mask.sum()
            print(f"{label}: {label_accuracy:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix (counts)")
    plt.tight_layout()
    plt.show()

    return report


def load_and_evaluate_model(
    joblib_path: str,
    X: pd.DataFrame,
    y_true,
    model_name: str = "Model",
    device=None,
    batch_size: int = 1024,
):
    joblib_object = joblib.load(joblib_path)

    model = joblib_object["model"]
    ordinal_encoder = joblib_object.get("ordinal_encoder")
    scaler = joblib_object.get("scaler")
    label_encoder = joblib_object.get("label_encoder")
    features = joblib_object.get("features")
    categorical_cols = joblib_object.get("categorical_cols")

    X = X.copy()

    if features:
        X = X[features]

    if categorical_cols:
        X[categorical_cols] = ordinal_encoder.transform(X[categorical_cols])

    X_scaled_df = X.copy()

    if scaler is not None:
        X_scaled_df[features] = scaler.transform(X[features])

        binary_cols = [c for c in ["valid_tcp_handshake_feature", "is_udp", "is_http"] if c in features]
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
            device = next(model.parameters()).device

        X_tensor = torch.tensor(X_np).unsqueeze(1)

        preds_all = []
        model.eval()

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                xb = X_tensor[i:i+batch_size].to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_all.extend(preds)

        y_pred = label_encoder.inverse_transform(np.array(preds_all))

    evaluate_model(y_true, y_pred, model_name=model_name)
