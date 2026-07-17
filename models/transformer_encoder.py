import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        p = torch.arange(max_len).unsqueeze(1); d = torch.arange(0, dim, 2)
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(p / (10000 ** (d / dim)))
        pe[:, 1::2] = torch.cos(p / (10000 ** (d / dim)))
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=128, heads=4, dropout=0.2):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim); self.pos = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.proj = nn.Linear(hidden_dim, embedding_dim)
    def forward(self, x):
        seq = self.encoder(self.pos(self.input(x)))
        return self.proj(seq.mean(1)), seq

