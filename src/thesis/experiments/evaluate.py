from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import property_driven_ml.training as pml_training
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from thesis.data.datasets import cross_eval_name
from thesis.data.preprocessing import transform_baseline_cross_eval, transform_property_cross_eval
from thesis.properties.constraints import build_constraints
from thesis.results.metrics import classification_outputs, save_eval_outputs
from thesis.results.plotting import plot_eval_summary
from thesis.training.baseline import predict_torch
from thesis.training.properties import PropertyTrainingContext, build_logic, evaluate_property_model, print_rule_stats


def _load_config(run_dir: Path, payload: dict) -> dict:
    if "config" in payload:
        return payload["config"]

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found in model payload or at {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _cross_data_paths(config: dict, cross_data: list[Path] | None) -> list[Path]:
    paths = cross_data or [Path(path) for path in config["data"].get("cross_eval_path", [])]
    if not paths:
        raise ValueError("No cross dataset provided. Pass --cross-data or set data.cross_eval_path in the saved config.")
    return paths


def _required(payload: dict, key: str):
    if key not in payload:
        raise ValueError(
            f"Saved model payload is missing {key!r}. "
            "Exact post-hoc evaluation requires models saved after evaluation metadata was added."
        )
    return payload[key]


def _unique_eval_names(paths: list[Path]) -> list[str]:
    used_names: set[str] = set()
    return [cross_eval_name(path, used_names) for path in paths]


def _baseline_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(x).unsqueeze(1), torch.tensor(y)),
        batch_size=batch_size,
    )


def _evaluate_baseline(run_dir: Path, payload: dict, config: dict, paths: list[Path]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = payload["model"]
    labels = payload["labels"]
    features = payload["features"]
    scaler = payload["scaler"]
    scale_cols = _required(payload, "scale_cols")
    clip_lower = _required(payload, "clip_lower")
    clip_upper = _required(payload, "clip_upper")
    names = _unique_eval_names(paths)

    for name, path in zip(names, paths):
        cross_eval = transform_baseline_cross_eval(
            path=path,
            config=config,
            features=features,
            scaler=scaler,
            scale_cols=scale_cols,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        if payload["model_type"] == "random_forest":
            y_pred = model.predict(cross_eval.x)
        else:
            loader = _baseline_loader(cross_eval.x, cross_eval.y, config["model"]["batch_size"])
            y_pred = predict_torch(model, loader, device)

        metrics, report_df, cm = classification_outputs(cross_eval.y, y_pred, labels)
        output_dir = run_dir / "cross_eval" / name
        save_eval_outputs(output_dir, metrics, report_df, cm, labels)
        plot_eval_summary(metrics, report_df, cm, labels, f"Baseline post-hoc cross eval: {name}", output_dir / "confusion_matrix.png")
        print(f"Saved baseline cross evaluation for {path} to: {output_dir}")


def _evaluate_properties(run_dir: Path, payload: dict, config: dict, paths: list[Path]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = payload["model"].to(device)
    labels = payload["labels"]
    tensor_features = payload["tensor_features"]
    scale_cols = payload["scale_cols"]
    scaler = payload["scaler"]
    clip_lower = payload["clip_lower"]
    clip_upper = payload["clip_upper"]
    model_feature_count = len(payload["features"])
    prop_cfg = config["properties"]
    logic = build_logic(prop_cfg.get("logic", "dl2"))
    oracle = pml_training.PGD(
        logic,
        device,
        steps=prop_cfg["pgd_steps"],
        restarts=prop_cfg["pgd_restarts"],
        step_size=prop_cfg["pgd_step_size"],
    )
    constraints = build_constraints(
        feature_cols=tensor_features,
        labels=labels,
        attack_specs=config["attack_specs"],
        preconditions=config["preconditions"],
        device=device,
        scaler=scaler,
        scale_cols=scale_cols,
        model_feature_count=model_feature_count,
    )
    ctx = PropertyTrainingContext(
        logic=logic,
        constraints=constraints,
        oracle=oracle,
        labels=labels,
        device=device,
        model_feature_count=model_feature_count,
        lambda_dos=prop_cfg["lambda_dos"],
        lambda_scan=prop_cfg["lambda_scan"],
    )
    names = _unique_eval_names(paths)

    for name, path in zip(names, paths):
        cross_eval = transform_property_cross_eval(
            path=path,
            config=config,
            tensor_features=tensor_features,
            scaler=scaler,
            scale_cols=scale_cols,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        cross_metrics, y_true, y_pred = evaluate_property_model(
            model,
            cross_eval.loader,
            ctx,
            collect_debug_stats=True,
        )
        dos_debug_stats = cross_metrics.pop("dos_debug_stats", {})
        scan_debug_stats = cross_metrics.pop("scan_debug_stats", {})
        metrics, report_df, cm = classification_outputs(y_true, y_pred, labels)
        metrics.update(cross_metrics)
        output_dir = run_dir / "cross_eval" / name
        save_eval_outputs(output_dir, metrics, report_df, cm, labels)
        plot_eval_summary(metrics, report_df, cm, labels, f"Property cross eval: {name}", output_dir / "confusion_matrix.png")
        print(
            f"\n----- CROSS EVAL: {name} -----\n"
            f"attack_f1={metrics['attack_macro_f1']:.4f} "
            f"acc={metrics['acc']:.4f} "
            f"adv_dos_loss={metrics['adv_dos_loss']:.4f} "
            f"csec_dos={metrics['csec_dos']:.4f} "
            f"adv_scan_loss={metrics['adv_scan_loss']:.4f} "
            f"csec_scan={metrics['csec_scan']:.4f} "
            f"csat_dos={metrics['csat_dos']:.4f} "
            f"csat_scan={metrics['csat_scan']:.4f}"
        )
        print(f"Saved property cross evaluation for {path} to: {output_dir}")
        print_rule_stats("DoS HTTP Flood", dos_debug_stats)
        print_rule_stats("Portscan", scan_debug_stats)


def evaluate_run(model_path: Path, cross_data: list[Path] | None = None) -> None:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected model file at {model_path}")
    if model_path.is_dir():
        raise IsADirectoryError(f"--model must point to a model.joblib file, not a run directory: {model_path}")

    run_dir = model_path.parent

    payload = joblib.load(model_path)
    config = _load_config(run_dir, payload)
    paths = _cross_data_paths(config, cross_data)
    if "tensor_features" in payload:
        _evaluate_properties(run_dir, payload, config, paths)
    else:
        _evaluate_baseline(run_dir, payload, config, paths)


def evaluate_tree(root: Path, cross_data: list[Path], pattern: str = "model.joblib") -> None:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Expected model root directory at {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"--root must point to a directory: {root}")

    model_paths = sorted(root.rglob(pattern))
    if not model_paths:
        raise FileNotFoundError(f"No {pattern!r} files found below {root}")

    print(f"Found {len(model_paths)} model(s) below {root}")
    for index, model_path in enumerate(model_paths, start=1):
        print(f"\n[{index}/{len(model_paths)}] Evaluating {model_path}")
        evaluate_run(model_path, cross_data)
