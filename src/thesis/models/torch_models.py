from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_features: int, num_classes: int, hidden_width: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(n_features, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.head(x)


def build_model(model_type: str, n_features: int, num_classes: int) -> nn.Module:
    hidden_widths = {
        "mlp": 64,
        "mlp_43k": 200,
        "mlp_186k": 422,
    }
    try:
        hidden_width = hidden_widths[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown torch model type: {model_type}") from exc
    return MLP(n_features=n_features, num_classes=num_classes, hidden_width=hidden_width)
