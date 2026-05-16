from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from thesis.data.datasets import load_experiment_data
from thesis.data.features import baseline_features
from thesis.data.preprocessing import fit_baseline_data, torch_loaders_from_arrays
from thesis.experiments.common import make_run_dir, save_json, save_model, save_run_config, set_seed
from thesis.models.torch_models import build_model
from thesis.results.metrics import classification_outputs, save_eval_outputs
from thesis.results.plotting import plot_eval_summary
from thesis.training.torch import predict_torch, train_torch_classifier


def _array_loader(x, y, batch_size):
    return DataLoader(TensorDataset(torch.tensor(x).unsqueeze(1), torch.tensor(y)), batch_size=batch_size)


def print_split_counts(name: str, labels: list[str], y) -> None:
    y = np.asarray(y)
    counts = {label: int((y == i).sum()) for i, label in enumerate(labels)}
    print(f"{name:10s} rows={len(y):8d} class_counts={counts}")


def print_metric_summary(name: str, metrics: dict) -> None:
    print(
        f"\n----- {name.upper()} -----\n"
        f"acc={metrics['acc']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


def run_baseline(config: dict):
    set_seed(config["experiment"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(config)
    save_run_config(config, run_dir)
    model_type = config["model"]["type"]

    print(
        f"experiment={config['experiment']['name']}\n"
        f"task={config['experiment'].get('task')} "
        f"model={model_type}\n"
        f"train_path={config['data']['train_path']}\n"
        f"cross_eval_path={config['data'].get('cross_eval_path')}"
    )

    features, categorical_cols, continuous_cols = baseline_features()
    data = fit_baseline_data(load_experiment_data(config), config, features, categorical_cols, continuous_cols)
    print(
        f"\nfeatures={len(data.features)} "
        f"categorical={len(data.categorical_cols)} "
        f"continuous={len(data.continuous_cols)}"
    )
    print_split_counts("train", data.labels, data.y_train)
    print_split_counts("val", data.labels, data.y_val)
    print_split_counts("test", data.labels, data.y_test)
    if data.y_cross_eval is not None:
        print_split_counts("cross_eval", data.labels, data.y_cross_eval)

    if model_type == "random_forest":
        n_estimators = config["model"].get("n_estimators", 100)
        n_jobs = config["model"].get("n_jobs", 1)
        seed = config["experiment"].get("seed", 42)
        print(
            f"\nrandom_forest n_estimators={n_estimators} "
            f"n_jobs={n_jobs} seed={seed}"
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=n_jobs,
        )
        print("fitting random forest...")
        model.fit(data.x_train, data.y_train)
        train_acc = float((model.predict(data.x_train) == data.y_train).mean())
        print(f"train_acc={train_acc:.4f}")
        print("predicting test...")
        y_pred = model.predict(data.x_test)
        cross_pred = None
        if data.x_cross_eval is not None:
            print("predicting cross_eval...")
            cross_pred = model.predict(data.x_cross_eval)
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
    print_metric_summary("test", metrics)
    save_eval_outputs(run_dir / "test", metrics, report_df, cm, data.labels)
    plot_eval_summary(metrics, report_df, cm, data.labels, "Baseline test", run_dir / "test" / "confusion_matrix.png")
    summary_metrics = {"test": metrics}

    if cross_pred is not None:
        metrics, report_df, cm = classification_outputs(data.y_cross_eval, cross_pred, data.labels)
        print_metric_summary("cross eval", metrics)
        save_eval_outputs(run_dir / "cross_eval", metrics, report_df, cm, data.labels)
        plot_eval_summary(metrics, report_df, cm, data.labels, "Baseline cross eval", run_dir / "cross_eval" / "confusion_matrix.png")
        summary_metrics["cross_eval"] = metrics

    save_json(run_dir / "metrics.json", summary_metrics)

    print(f"Run saved to: {run_dir}")
    return run_dir
