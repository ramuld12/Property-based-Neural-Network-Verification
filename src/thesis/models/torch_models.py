from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_features: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.head(x)


def build_model(model_type: str, n_features: int, num_classes: int) -> nn.Module:
    return MLP(n_features=n_features, num_classes=num_classes)
