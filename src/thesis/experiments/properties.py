from __future__ import annotations

import torch

from thesis.data.datasets import load_experiment_data
from thesis.data.features import SHARED_MODEL_FEATURES
from thesis.data.preprocessing import fit_property_data
from thesis.experiments.common import make_run_dir, save_json, save_model, save_run_config, set_seed
from thesis.models.torch_models import build_model
from thesis.properties.constraints import build_constraints
from thesis.results.metrics import classification_outputs, save_eval_outputs
from thesis.results.plotting import plot_eval_summary
from thesis.training.properties import evaluate_property_model, print_rule_stats, train_property_classifier


def run_properties(config: dict):
    set_seed(config["experiment"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(config)
    save_run_config(config, run_dir)

    feature_cols = list(SHARED_MODEL_FEATURES)
    data = fit_property_data(load_experiment_data(config), config, feature_cols)
    constraints = build_constraints(
        feature_cols=data.tensor_features,
        labels=data.labels,
        attack_specs=data.attack_specs,
        preconditions=config["preconditions"],
        device=device,
        scaler=data.scaler,
        scale_cols=data.scale_cols,
        model_feature_count=data.model_feature_count,
    )
    model = build_model(config["model"]["type"], n_features=len(data.features), num_classes=len(data.labels))
    model, history, ctx, best_epoch, best_score = train_property_classifier(model, data, constraints, config, device)

    history.to_csv(run_dir / "training_history.csv", index=False)
    save_json(run_dir / "metrics.json", {"best_epoch": best_epoch, "best_score": best_score})
    save_model(
        run_dir / "model.joblib",
        {
            "model": model.cpu(),
            "features": data.features,
            "tensor_features": data.tensor_features,
            "labels": data.labels,
            "model_type": config["model"]["type"],
            "scaler": data.scaler,
            "scale_cols": data.scale_cols,
            "clip_lower": data.clip_lower,
            "clip_upper": data.clip_upper,
            "config": config,
        },
    )
    model = model.to(device)

    test_metrics, y_true, y_pred = evaluate_property_model(model, data.test_loader, ctx)
    metrics, report_df, cm = classification_outputs(y_true, y_pred, data.labels)
    metrics.update(test_metrics)
    save_eval_outputs(run_dir / "test", metrics, report_df, cm, data.labels)
    plot_eval_summary(metrics, report_df, cm, data.labels, "Property model test", run_dir / "test" / "confusion_matrix.png")

    for cross_eval in data.cross_evals:
        cross_metrics, y_true, y_pred = evaluate_property_model(
            model,
            cross_eval.loader,
            ctx,
            collect_debug_stats=True,
        )
        dos_debug_stats = cross_metrics.pop("dos_debug_stats", {})
        scan_debug_stats = cross_metrics.pop("scan_debug_stats", {})
        metrics, report_df, cm = classification_outputs(y_true, y_pred, data.labels)
        metrics.update(cross_metrics)
        print(
            f"\n----- CROSS EVAL: {cross_eval.name} -----\n"
            f"attack_f1={metrics['attack_macro_f1']:.4f} "
            f"acc={metrics['acc']:.4f} "
            f"adv_dos_loss={metrics['adv_dos_loss']:.4f} "
            f"csec_dos={metrics['csec_dos']:.4f} "
            f"adv_scan_loss={metrics['adv_scan_loss']:.4f} "
            f"csec_scan={metrics['csec_scan']:.4f}"
        )
        print_rule_stats("DoS HTTP Flood", dos_debug_stats)
        print_rule_stats("Portscan", scan_debug_stats)
        cross_eval_dir = run_dir / "cross_eval" if cross_eval.name == "cross_eval" else run_dir / "cross_eval" / cross_eval.name
        save_eval_outputs(cross_eval_dir, metrics, report_df, cm, data.labels)
        plot_eval_summary(metrics, report_df, cm, data.labels, f"Property model cross eval: {cross_eval.name}", cross_eval_dir / "confusion_matrix.png")

    print(f"Run saved to: {run_dir}")
    return run_dir
