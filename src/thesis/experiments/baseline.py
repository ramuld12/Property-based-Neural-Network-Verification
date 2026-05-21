from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from thesis.data.datasets import load_experiment_data
from thesis.data.features import SHARED_MODEL_FEATURES
from thesis.data.preprocessing import fit_baseline_data, torch_loaders_from_arrays
from thesis.experiments.common import make_run_dir, save_json, save_model, save_run_config, set_seed
from thesis.models.torch_models import build_model
from thesis.results.metrics import classification_outputs, save_eval_outputs
from thesis.results.plotting import plot_eval_summary
from thesis.training.baseline import predict_torch, train_torch_classifier


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

    features = list(SHARED_MODEL_FEATURES)
    data = fit_baseline_data(load_experiment_data(config), config, features)
    print(f"\nfeatures={len(data.features)}")
    print_split_counts("train", data.labels, data.y_train)
    print_split_counts("val", data.labels, data.y_val)
    print_split_counts("test", data.labels, data.y_test)
    for cross_eval in data.cross_evals:
        print_split_counts(f"cross:{cross_eval.name}", data.labels, cross_eval.y)

    runtime_metrics = None
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
        fit_start = time.perf_counter()
        model.fit(data.x_train, data.y_train)
        fit_seconds = time.perf_counter() - fit_start
        runtime_metrics = {"fit_seconds": fit_seconds}
        print(f"fit_time={fit_seconds:.2f}s")
        train_acc = float((model.predict(data.x_train) == data.y_train).mean())
        print(f"train_acc={train_acc:.4f}")
        print("predicting test...")
        y_pred = model.predict(data.x_test)
        cross_preds = []
        for cross_eval in data.cross_evals:
            print(f"predicting cross_eval: {cross_eval.name}...")
            cross_preds.append((cross_eval, model.predict(cross_eval.x)))
    else:
        train_loader, val_loader, test_loader = torch_loaders_from_arrays(data, config["model"]["batch_size"])
        model = build_model(model_type, n_features=len(data.features), num_classes=len(data.labels))
        model, history = train_torch_classifier(model, train_loader, val_loader, config, device)
        history.to_csv(run_dir / "training_history.csv", index=False)
        y_pred = predict_torch(model, test_loader, device)
        cross_preds = []
        for cross_eval in data.cross_evals:
            loader = _array_loader(cross_eval.x, np.zeros(len(cross_eval.x), dtype=np.int64), config["model"]["batch_size"])
            cross_preds.append((cross_eval, predict_torch(model, loader, device)))

    save_model(
        run_dir / "model.joblib",
        {
            "model": model.cpu() if hasattr(model, "cpu") else model,
            "features": data.features,
            "labels": data.labels,
            "model_type": model_type,
            "scaler": data.scaler,
            "config": config,
        },
    )

    metrics, report_df, cm = classification_outputs(data.y_test, y_pred, data.labels)
    print_metric_summary("test", metrics)
    save_eval_outputs(run_dir / "test", metrics, report_df, cm, data.labels)
    plot_eval_summary(metrics, report_df, cm, data.labels, "Baseline test", run_dir / "test" / "confusion_matrix.png")
    summary_metrics = {"test": metrics}
    if runtime_metrics is not None:
        summary_metrics["runtime"] = runtime_metrics

    for cross_eval, cross_pred in cross_preds:
        metrics, report_df, cm = classification_outputs(cross_eval.y, cross_pred, data.labels)
        print_metric_summary(f"cross eval: {cross_eval.name}", metrics)
        cross_eval_dir = run_dir / "cross_eval" if cross_eval.name == "cross_eval" else run_dir / "cross_eval" / cross_eval.name
        save_eval_outputs(cross_eval_dir, metrics, report_df, cm, data.labels)
        plot_eval_summary(metrics, report_df, cm, data.labels, f"Baseline cross eval: {cross_eval.name}", cross_eval_dir / "confusion_matrix.png")
        if cross_eval.name == "cross_eval":
            summary_metrics["cross_eval"] = metrics
        else:
            summary_metrics.setdefault("cross_eval", {})[cross_eval.name] = metrics

    save_json(run_dir / "metrics.json", summary_metrics)

    print(f"Run saved to: {run_dir}")
    return run_dir
