from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def improved_attack_f1_or_loss(attack_f1: float, best_attack_f1: float, val_loss: float, best_loss: float, min_delta: float) -> bool:
    if attack_f1 > best_attack_f1 + min_delta:
        return True
    tied_attack_f1 = abs(attack_f1 - best_attack_f1) <= min_delta
    return tied_attack_f1 and val_loss < best_loss - min_delta


def train_torch_classifier(model, train_loader, val_loader, config: dict, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    patience = config["model"].get("patience", 5)
    min_delta = config["model"].get("min_delta", 1e-4)
    labels = config["data"]["labels"]
    attack_ids = list(range(1, len(labels)))
    best_attack_f1 = -float("inf")
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["model"]["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        train_true, train_pred = [], []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_true.extend(y.cpu().numpy())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().numpy())

        val_losses = []
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                val_losses.append(criterion(logits, y).item())
                y_true.extend(y.cpu().numpy())
                y_pred.extend(logits.argmax(dim=1).cpu().numpy())

        train_loss = float(np.mean(train_losses))
        train_acc = float((np.asarray(train_true) == np.asarray(train_pred)).mean())
        val_loss = float(np.mean(val_losses))
        val_acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
        val_attack_f1 = float(f1_score(y_true, y_pred, labels=attack_ids, average="macro", zero_division=0))
        epoch_seconds = time.perf_counter() - epoch_start
        improved = improved_attack_f1_or_loss(val_attack_f1, best_attack_f1, val_loss, best_loss, min_delta)
        if improved:
            best_attack_f1 = val_attack_f1
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_attack_macro_f1": val_attack_f1,
            "epoch_seconds": epoch_seconds,
        })
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_attack_f1={val_attack_f1:.4f} "
            f"epoch_time={epoch_seconds:.2f}s "
            f"patience={epochs_without_improvement}/{patience}"
        )

        if epochs_without_improvement >= patience:
            print(f"early stopping at epoch={epoch} best_val_attack_f1={best_attack_f1:.4f} best_val_loss={best_loss:.4f}")
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_torch(model, loader, device):
    model = model.to(device)
    model.eval()
    preds = []
    for x, _ in loader:
        preds.extend(model(x.to(device)).argmax(dim=1).cpu().numpy())
    return np.asarray(preds)
