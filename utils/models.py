import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, n_features: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: [B, 1, n_features]
        x = self.conv(x)              # [B, 128, L]
        x = x.transpose(1, 2)         # [B, L, 128]
        _, (h_n, _) = self.lstm(x)    # h_n: [1, B, 64]
        x = h_n[-1]                   # [B, 64]
        x = self.head(x)              # [B, num_classes]
        return x

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
        # x: [B, 1, n_features]
        x = x.squeeze(1)          # -> [B, n_features]
        x = self.head(x)         # -> [B, num_classes]
        return x
    
def build_model(model_type: str, n_features: int, num_classes: int) -> nn.Module:
    model_type = model_type.lower()

    if model_type == "mlp":
        return MLP(
            n_features=n_features,
            num_classes=num_classes,
        )
    elif model_type == "cnnlstm":
        return CNNLSTM(
            n_features=n_features,
            num_classes=num_classes,
        )