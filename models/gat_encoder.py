import torch
from torch import nn
import torch.nn.functional as F

class DenseEdgeGATLayer(nn.Module):
    """Small-batch multi-head GAT without torch-geometric."""
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        if out_dim % heads: raise ValueError("out_dim must be divisible by heads")
        self.heads, self.d = heads, out_dim // heads
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.d)); self.a_dst = nn.Parameter(torch.empty(heads, self.d))
        self.bias = nn.Parameter(torch.zeros(out_dim)); self.dropout = dropout
        nn.init.xavier_uniform_(self.linear.weight); nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
    def forward(self, x, edge_index):
        n = x.size(0); h = self.linear(x).view(n, self.heads, self.d)
        src, dst = edge_index
        e = F.leaky_relu((h[src] * self.a_src).sum(-1) + (h[dst] * self.a_dst).sum(-1), 0.2)
        out = torch.zeros_like(h)
        for node in range(n):
            mask = dst == node
            if mask.any():
                alpha = F.dropout(torch.softmax(e[mask], dim=0), self.dropout, self.training)
                out[node] = (alpha.unsqueeze(-1) * h[src[mask]]).sum(0)
            else: out[node] = h[node]
        return out.reshape(n, -1) + self.bias

class GATEncoder(nn.Module):
    def __init__(self, dim=128, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = DenseEdgeGATLayer(dim, dim, heads, dropout)
        self.gat2 = DenseEdgeGATLayer(dim, dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim); self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.norm1(x + self.dropout(F.elu(self.gat1(x, edge_index))))
        return self.norm2(x + self.dropout(self.gat2(x, edge_index)))

