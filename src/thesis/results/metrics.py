from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def classification_outputs(y_true, y_pred, labels: list[str]) -> tuple[dict, pd.DataFrame, np.ndarray]:
    label_ids = list(range(len(labels)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    for label_id, label_name in zip(label_ids, labels):
        mask = np.asarray(y_true) == label_id
        report[label_name]["accuracy"] = float((np.asarray(y_pred)[mask] == label_id).mean()) if mask.sum() else 0.0
    metrics = {
        "acc": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics, pd.DataFrame(report).T, confusion_matrix(y_true, y_pred, labels=label_ids)


def save_eval_outputs(output_dir: Path, metrics: dict, report_df: pd.DataFrame, cm: np.ndarray, labels: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_df.to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")
