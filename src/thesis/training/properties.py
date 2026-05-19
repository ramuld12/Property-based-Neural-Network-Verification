from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import property_driven_ml.logics as logics
import property_driven_ml.training as pml_training
from sklearn.metrics import f1_score

from thesis.data.preprocessing import DEBUG_LABEL_DOS_HTTP_FLOOD, DEBUG_LABEL_PORTSCAN


@dataclass
class PropertyTrainingContext:
    logic: object
    constraints: dict
    oracle: object
    labels: list[str]
    device: torch.device
    model_feature_count: int
    lambda_dos: float
    lambda_scan: float


def make_weighted_ce_loss(train_df: pd.DataFrame, device: torch.device) -> nn.CrossEntropyLoss:
    class_counts = train_df["label_id"].value_counts().sort_index().to_numpy()
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = (class_weights / class_weights.mean()).to(device)
    print("class_counts:", class_counts)
    print("class_weights:", class_weights)
    return nn.CrossEntropyLoss(weight=class_weights)


def build_logic(name: str):
    name = name.lower()
    logic_classes = {
        "goedel": logics.GoedelFuzzyLogic,
        "boolean": logics.BooleanLogic,
        "dl2": logics.DL2,
        "lukasiewicz": logics.LukasiewiczFuzzyLogic,
        "reichenbach": logics.ReichenbachFuzzyLogic,
        "yager": logics.YagerFuzzyLogic,
        "stl": logics.STL,
        "qll": logics.QLL,
    }
    try:
        return logic_classes[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown logic: {name}") from exc


def precondition_bounds(precondition, x):
    if hasattr(precondition, "get_bounds"):
        return precondition.get_bounds(x)
    return precondition.get_precondition(x)


def project_adv_if_global(x_adv, x, constraint):
    lo, hi = precondition_bounds(constraint.precondition, x)
    return torch.max(torch.min(x_adv, hi), lo)


def make_consistent_adversarial(model, oracle, x, constraint):
    x_adv = oracle.attack(model, x, None, constraint)
    return project_adv_if_global(x_adv, x, constraint)


def update_rule_stats(stats: dict, parts: dict) -> None:
    for name, mask in parts.items():
        filtered_mask = mask.detach().bool()
        stats.setdefault(name, {"true": 0, "total": 0})
        stats[name]["true"] += filtered_mask.sum().item()
        stats[name]["total"] += filtered_mask.numel()


def print_rule_stats(name: str, stats: dict) -> None:
    if not stats:
        return
    print(f"\n{name} rule debug - original-subtype rows only")
    for part_name, values in stats.items():
        true = values["true"]
        total = values["total"]
        print(f"{part_name:25s} {true:8d}/{total:<8d}  {true / max(total, 1):.4f}")


def update_debug_stats_for_subtype(stats: dict, constraint, model, x, x_adv, debug_y, subtype_id: int) -> None:
    mask = debug_y == subtype_id
    if mask.any():
        update_rule_stats(stats, constraint.postcondition.debug_parts(model, x[mask], x_adv[mask]))


def train_one_epoch(model, optimizer, grad_norm, oracle, ce_fn, loader, ctx: PropertyTrainingContext, epoch: int) -> dict:
    model.train()
    totals = {"ce_loss": 0.0, "scaled_dos_loss": 0.0, "scaled_scan_loss": 0.0, "dos_sat": 0.0, "scan_sat": 0.0}
    dos_debug_stats, scan_debug_stats = {}, {}
    for x, y, debug_y in loader:
        x = x.to(ctx.device)
        y = y.to(ctx.device)
        debug_y = debug_y.to(ctx.device)
        optimizer.zero_grad()
        ce_loss = ce_fn(model(x[:, : ctx.model_feature_count]), y)
        constraint_loss = torch.tensor(0.0, device=ctx.device)

        x_adv_dos = make_consistent_adversarial(model, oracle, x, ctx.constraints["dos"])
        dos_loss, dos_sat = ctx.constraints["dos"].eval(model, x, x_adv_dos, None, ctx.logic, reduction="mean")
        totals["scaled_dos_loss"] += ctx.lambda_dos * dos_loss.item()
        totals["dos_sat"] += dos_sat.item()
        update_debug_stats_for_subtype(
            dos_debug_stats,
            ctx.constraints["dos"],
            model,
            x,
            x_adv_dos,
            debug_y,
            DEBUG_LABEL_DOS_HTTP_FLOOD,
        )

        x_adv_scan = make_consistent_adversarial(model, oracle, x, ctx.constraints["scan"])
        scan_loss, scan_sat = ctx.constraints["scan"].eval(model, x, x_adv_scan, None, ctx.logic, reduction="mean")
        constraint_loss = ctx.lambda_dos * dos_loss + ctx.lambda_scan * scan_loss
        totals["scaled_scan_loss"] += ctx.lambda_scan * scan_loss.item()
        totals["scan_sat"] += scan_sat.item()
        update_debug_stats_for_subtype(
            scan_debug_stats,
            ctx.constraints["scan"],
            model,
            x,
            x_adv_scan,
            debug_y,
            DEBUG_LABEL_PORTSCAN,
        )
        grad_norm.balance(ce_loss, constraint_loss)

        totals["ce_loss"] += ce_loss.item()

    metrics = {key: value / len(loader) for key, value in totals.items()}
    metrics["dos_debug_stats"] = dos_debug_stats
    metrics["scan_debug_stats"] = scan_debug_stats
    return metrics


def evaluate_property_model(
    model,
    loader,
    ctx: PropertyTrainingContext,
    ce_fn=None,
    collect_debug_stats: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    totals = {"adv_dos_loss": 0.0, "adv_scan_loss": 0.0, "adv_dos_sat": 0.0, "adv_scan_sat": 0.0}
    counts = {"dos": 0, "scan": 0}
    ce_losses = []
    dos_debug_stats, scan_debug_stats = {}, {}

    for x, y, debug_y in loader:
        x = x.to(ctx.device)
        y = y.to(ctx.device)
        debug_y = debug_y.to(ctx.device)
        with torch.no_grad():
            logits = model(x[:, : ctx.model_feature_count])
            preds = logits.argmax(dim=1)
            if ce_fn is not None:
                ce_losses.append(ce_fn(logits, y).item())
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        x_adv_dos = make_consistent_adversarial(model, ctx.oracle, x, ctx.constraints["dos"])
        with torch.no_grad():
            loss, sat = ctx.constraints["dos"].eval(model, x, x_adv_dos, None, ctx.logic, reduction="sum")
        totals["adv_dos_loss"] += loss.item()
        totals["adv_dos_sat"] += sat.item()
        counts["dos"] += x.size(0)
        if collect_debug_stats:
            update_debug_stats_for_subtype(
                dos_debug_stats,
                ctx.constraints["dos"],
                model,
                x,
                x_adv_dos,
                debug_y,
                DEBUG_LABEL_DOS_HTTP_FLOOD,
            )

        x_adv_scan = make_consistent_adversarial(model, ctx.oracle, x, ctx.constraints["scan"])
        with torch.no_grad():
            loss, sat = ctx.constraints["scan"].eval(model, x, x_adv_scan, None, ctx.logic, reduction="sum")
        totals["adv_scan_loss"] += loss.item()
        totals["adv_scan_sat"] += sat.item()
        counts["scan"] += x.size(0)
        if collect_debug_stats:
            update_debug_stats_for_subtype(
                scan_debug_stats,
                ctx.constraints["scan"],
                model,
                x,
                x_adv_scan,
                debug_y,
                DEBUG_LABEL_PORTSCAN,
            )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    attack_ids = list(range(1, len(ctx.labels)))
    metrics = {
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "attack_macro_f1": float(f1_score(y_true, y_pred, labels=attack_ids, average="macro", zero_division=0)),
        "adv_dos_loss": totals["adv_dos_loss"] / max(counts["dos"], 1),
        "adv_scan_loss": totals["adv_scan_loss"] / max(counts["scan"], 1),
        "adv_dos_sat": totals["adv_dos_sat"] / max(counts["dos"], 1),
        "adv_scan_sat": totals["adv_scan_sat"] / max(counts["scan"], 1),
    }
    if ce_fn is not None:
        metrics["ce_loss"] = float(np.mean(ce_losses))
    if collect_debug_stats:
        metrics["dos_debug_stats"] = dos_debug_stats
        metrics["scan_debug_stats"] = scan_debug_stats
    return metrics, y_true, y_pred


def model_selection_score(metrics: dict) -> float:
    return 2.0 * metrics["attack_macro_f1"] + 0.5 * metrics["macro_f1"] + 0.5 * metrics["adv_dos_sat"] + 0.5 * metrics["adv_scan_sat"] + 0.5 * metrics["acc"]


def train_property_classifier(model, data, constraints: dict, config: dict, device):
    model = model.to(device)
    prop_cfg = config["properties"]
    logic = build_logic(prop_cfg.get("logic", "dl2"))
    oracle = pml_training.PGD(logic, device, steps=prop_cfg["pgd_steps"], restarts=prop_cfg["pgd_restarts"], step_size=prop_cfg["pgd_step_size"])
    ctx = PropertyTrainingContext(
        logic=logic,
        constraints=constraints,
        oracle=oracle,
        labels=data.labels,
        device=device,
        model_feature_count=data.model_feature_count,
        lambda_dos=prop_cfg["lambda_dos"],
        lambda_scan=prop_cfg["lambda_scan"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    grad_norm = pml_training.GradNorm(model, device, optimizer, lr=config["model"]["learning_rate"], alpha=1.5)
    ce_fn = make_weighted_ce_loss(data.train_df, device)

    best_score = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    patience = config["model"].get("patience", 5)
    min_delta = config["model"].get("min_delta", 1e-4)
    min_epochs = config["model"].get("min_epochs", 1)

    for epoch in range(1, config["model"]["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, optimizer, grad_norm, oracle, ce_fn, data.train_loader, ctx, epoch)
        val_metrics, _, _ = evaluate_property_model(model, data.val_loader, ctx, ce_fn=ce_fn)
        epoch_seconds = time.perf_counter() - epoch_start
        score = model_selection_score(val_metrics)
        improved = score > best_score + min_delta
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items() if not k.endswith("_stats")},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "selection_score": score,
            "epoch_seconds": epoch_seconds,
        })
        print( 
            f"epoch={epoch} \n" 
            f"----- TRAIN -----\n" 
            f"ce_loss={train_metrics['ce_loss']:.4f} " 
            f"scaled_dos_loss={train_metrics['scaled_dos_loss']:.4f} " 
            f"scaled_scan_loss={train_metrics['scaled_scan_loss']:.4f} " 
            f"dos_sat={train_metrics['dos_sat']:.4f} " 
            f"scan_sat={train_metrics['scan_sat']:.4f} \n " 
            f"----- VALIDATION -----\n" f"val_ce_loss={val_metrics['ce_loss']:.4f} "
            f"attack_f1={val_metrics['attack_macro_f1']:.4f} " 
            f"acc={val_metrics['acc']:.4f} " 
            f"adv_dos_loss={val_metrics['adv_dos_loss']:.4f} " 
            f"adv_dos_sat={val_metrics['adv_dos_sat']:.4f} " 
            f"adv_scan_loss={val_metrics['adv_scan_loss']:.4f} " 
            f"adv_scan_sat={val_metrics['adv_scan_sat']:.4f} \n" 
            f"score={score:.4f} "
            f"best_score={best_score:.4f} "
            f"epoch_time={epoch_seconds:.2f}s "
            f"patience={epochs_without_improvement}/{patience}" 
        )
        print_rule_stats("DoS HTTP Flood", train_metrics["dos_debug_stats"])
        print_rule_stats("Portscan", train_metrics["scan_debug_stats"])

        if epoch >= min_epochs and epochs_without_improvement >= patience:
            print(
                f"early stopping at epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"best_score={best_score:.4f}"
            )
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), ctx, best_epoch, best_score
