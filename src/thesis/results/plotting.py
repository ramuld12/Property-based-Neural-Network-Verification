from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_eval_summary(metrics, report_df: pd.DataFrame, cm, labels: list[str], model_name: str, save_path: Path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 15), gridspec_kw={"height_ratios": [3, 1.2, 2.8]})
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title(f"{model_name} Confusion Matrix")

    ax2.axis("off")
    metric_lines = [f"{key}: {value:.3f}" for key, value in metrics.items() if isinstance(value, float | int)]
    ax2.text(0.01, 0.95, "\n".join(metric_lines), va="top", ha="left", family="monospace", fontsize=11)

    ax3.axis("off")
    ax3.text(0.01, 0.95, report_df.round(3).to_string(), va="top", ha="left", family="monospace", fontsize=9)
    ax3.set_title("Classification Report + Per-Label Accuracy")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
