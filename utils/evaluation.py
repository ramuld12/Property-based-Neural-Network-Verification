"""Model evaluation utilities."""

import numpy as np
import pandas as pd
import joblib
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


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


def load_and_evaluate_rf_model(
    joblib_path: str, X: pd.DataFrame, y_true, model_name: str = "Model"
):
    """Load and evaluate a Random Forest model.
    
    Args:
        joblib_path: Path to joblib model file
        X: Feature DataFrame
        y_true: True labels
        model_name: Name for display purposes
    """
    joblib_object = joblib.load(joblib_path)
    model = joblib_object["model"]
    encoder = joblib_object.get("encoder")

    # Encode categorical features
    X = X.copy()
    categorical_cols = X.select_dtypes(include=["object"]).columns
    if categorical_cols is not None and encoder is not None:
        X[categorical_cols] = encoder.transform(X[categorical_cols])
    
    y_pred = model.predict(X)
    print(f"Evaluation for {model_name}:")
    evaluate_model(y_true, y_pred, model_name=model_name)


def load_and_evaluate_cnnlstm_model(
    joblib_path: str, X: pd.DataFrame, y_true, model_name: str = "Model", device=None
):
    """Load and evaluate a CNN-LSTM model.
    
    Args:
        joblib_path: Path to joblib model file
        X: Feature DataFrame
        y_true: True labels
        model_name: Name for display purposes
        device: Torch device (auto-detected if None)
    """
    joblib_object = joblib.load(joblib_path)
    model = joblib_object["model"]
    ordinal_encoder = joblib_object.get("ordinal_encoder")
    scaler = joblib_object.get("scaler")
    label_encoder = joblib_object.get("label_encoder")
    features = joblib_object.get("features")
    categorical_cols = joblib_object.get("categorical_cols")

    # Determine device from model if not provided
    if device is None:
        device = next(model.parameters()).device

    X = X.copy()

    # Encode categorical features
    X[categorical_cols] = ordinal_encoder.transform(X[categorical_cols])

    # Scale features
    if scaler is not None:
        if features:
            X_scaled = scaler.transform(X[features])
        else:
            X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    # Convert to tensor and add channel dimension
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    y_pred_labels = label_encoder.inverse_transform(preds)


    print(f"Evaluation for {model_name}:")
    evaluate_model(y_true, y_pred_labels, model_name=model_name)
