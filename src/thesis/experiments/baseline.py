from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from thesis.data.datasets import load_experiment_data
from thesis.data.features import baseline_features
from thesis.data.preprocessing import fit_baseline_data, torch_loaders_from_arrays
from thesis.experiments.common import make_run_dir, save_model, save_run_config, set_seed
from thesis.models.torch_models import build_model
from thesis.results.metrics import classification_outputs, save_eval_outputs
from thesis.results.plotting import plot_eval_summary
from thesis.training.torch import predict_torch, train_torch_classifier


def _array_loader(x, y, batch_size):
    return DataLoader(TensorDataset(torch.tensor(x).unsqueeze(1), torch.tensor(y)), batch_size=batch_size)


def run_baseline(config: dict):
    set_seed(config["experiment"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(config)
    save_run_config(config, run_dir)

    features, categorical_cols, continuous_cols = baseline_features(config)
    data = fit_baseline_data(load_experiment_data(config), config, features, categorical_cols, continuous_cols)
    model_type = config["model"]["type"]

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config["model"].get("n_estimators", 100),
            random_state=config["experiment"].get("seed", 42),
            n_jobs=config["model"].get("n_jobs", 1),
        )
        model.fit(data.x_train, data.y_train)
        y_pred = model.predict(data.x_test)
        cross_pred = None if data.x_cross_eval is None else model.predict(data.x_cross_eval)
    else:
        train_loader, val_loader, test_loader = torch_loaders_from_arrays(data, config["model"]["batch_size"])
        model = build_model(model_type, n_features=len(data.features), num_classes=len(data.labels))
        model, history = train_torch_classifier(model, train_loader, val_loader, config, device)
        history.to_csv(run_dir / "training_history.csv", index=False)
        y_pred = predict_torch(model, test_loader, device)
        cross_pred = None
        if data.x_cross_eval is not None:
            loader = _array_loader(data.x_cross_eval, np.zeros(len(data.x_cross_eval), dtype=np.int64), config["model"]["batch_size"])
            cross_pred = predict_torch(model, loader, device)

    save_model(
        run_dir / "model.joblib",
        {
            "model": model.cpu() if hasattr(model, "cpu") else model,
            "features": data.features,
            "labels": data.labels,
            "model_type": model_type,
            "ordinal_encoder": data.ordinal_encoder,
            "scaler": data.scaler,
            "categorical_cols": data.categorical_cols,
            "continuous_cols": data.continuous_cols,
            "config": config,
        },
    )

    metrics, report_df, cm = classification_outputs(data.y_test, y_pred, data.labels)
    save_eval_outputs(run_dir / "test", metrics, report_df, cm, data.labels)
    plot_eval_summary(metrics, report_df, cm, data.labels, "Baseline test", run_dir / "test" / "confusion_matrix.png")

    if cross_pred is not None:
        metrics, report_df, cm = classification_outputs(data.y_cross_eval, cross_pred, data.labels)
        save_eval_outputs(run_dir / "cross_eval", metrics, report_df, cm, data.labels)
        plot_eval_summary(metrics, report_df, cm, data.labels, "Baseline cross eval", run_dir / "cross_eval" / "confusion_matrix.png")

    print(f"Run saved to: {run_dir}")
    return run_dir
