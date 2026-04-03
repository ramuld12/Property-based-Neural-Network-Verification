import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid model for sequence classification.
    
    Architecture:
    - 2x Conv1d layers with ReLU, BatchNorm, and MaxPool
    - 1x LSTM layer
    - 2x Linear layers with Dropout in classification head
    
    Args:
        n_features: Number of input features
    """

    def __init__(self, n_features: int):
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
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, n_features)
            
        Returns:
            Logits tensor of shape (batch_size,)
        """
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last).squeeze(1)
        return logits
