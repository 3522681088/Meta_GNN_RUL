import torch
from torch import nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=128, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x):
        sequence, (h, _) = self.lstm(x)
        return self.proj(h[-1]), sequence

