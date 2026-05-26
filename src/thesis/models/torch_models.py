from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_classes: int,
        hidden_width: int | tuple[int, ...] = 64,
    ):
        super().__init__()
        hidden_layers = (
            hidden_width if isinstance(hidden_width, tuple) else (hidden_width, hidden_width)
        )
        layers = []
        previous_width = n_features
        for width in hidden_layers:
            layers.extend([nn.Linear(previous_width, width), nn.ReLU()])
            previous_width = width
        layers.append(nn.Linear(previous_width, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.head(x)


def build_model(model_type: str, n_features: int, num_classes: int) -> nn.Module:
    hidden_widths = {
        "mlp": 64,
        "mlp_43k": 200,
        "mlp_186k": (304, 288, 224, 128),
        "mlp_449k": (512, 512, 256, 128),
    }
    try:
        hidden_width = hidden_widths[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown torch model type: {model_type}") from exc
    return MLP(n_features=n_features, num_classes=num_classes, hidden_width=hidden_width)
