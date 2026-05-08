from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def train_torch_classifier(model, train_loader, val_loader, config: dict, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    patience = config["model"].get("patience", 5)
    min_delta = config["model"].get("min_delta", 1e-4)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["model"]["epochs"] + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                val_losses.append(criterion(model(x), y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
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
