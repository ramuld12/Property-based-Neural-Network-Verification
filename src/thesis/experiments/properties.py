from __future__ import annotations

import torch

from thesis.data.datasets import load_experiment_data
from thesis.data.features import DEFAULT_PROPERTY_FROZEN_FEATURES, SHARED_MODEL_FEATURES
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
    frozen_features = config["properties"].get("frozen_features", DEFAULT_PROPERTY_FROZEN_FEATURES)
    constraints = build_constraints(
        feature_cols=data.tensor_features,
        labels=data.labels,
        scaled_attack_specs=data.scaled_attack_specs,
        preconditions=config["preconditions"],
        device=device,
        scaler=data.scaler,
        scale_cols=data.scale_cols,
        model_feature_count=data.model_feature_count,
        frozen_feature_names=frozen_features,
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
            "frozen_features": frozen_features,
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

    if data.cross_eval_loader is not None:
        cross_metrics, y_true, y_pred = evaluate_property_model(
            model,
            data.cross_eval_loader,
            ctx,
            collect_debug_stats=True,
        )
        dos_debug_stats = cross_metrics.pop("dos_debug_stats", {})
        scan_debug_stats = cross_metrics.pop("scan_debug_stats", {})
        metrics, report_df, cm = classification_outputs(y_true, y_pred, data.labels)
        metrics.update(cross_metrics)
        print(
            "\n----- CROSS EVAL -----\n"
            f"attack_f1={metrics['attack_macro_f1']:.4f} "
            f"acc={metrics['acc']:.4f} "
            f"adv_dos_loss={metrics['adv_dos_loss']:.4f} "
            f"adv_dos_sat={metrics['adv_dos_sat']:.4f} "
            f"adv_scan_loss={metrics['adv_scan_loss']:.4f} "
            f"adv_scan_sat={metrics['adv_scan_sat']:.4f}"
        )
        print_rule_stats("DoS HTTP Flood", dos_debug_stats)
        print_rule_stats("Portscan", scan_debug_stats)
        save_eval_outputs(run_dir / "cross_eval", metrics, report_df, cm, data.labels)
        plot_eval_summary(metrics, report_df, cm, data.labels, "Property model cross eval", run_dir / "cross_eval" / "confusion_matrix.png")

    print(f"Run saved to: {run_dir}")
    return run_dir
