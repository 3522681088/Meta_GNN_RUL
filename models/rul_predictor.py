import torch
from torch import nn

class RULPredictor(nn.Module):
    def __init__(self,input_dim,dropout=0.2):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(input_dim,256),nn.LeakyReLU(0.2),nn.Dropout(dropout),nn.Linear(256,1))
    def forward(self,x): return self.network(x).squeeze(-1)

class PairwiseDistancePredictor(nn.Module):
    def __init__(self,embedding_dim,dropout=0.2):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(embedding_dim*2,256),nn.LeakyReLU(0.2),nn.Dropout(dropout),nn.Linear(256,1))
    def forward(self,left,right): return self.network(torch.cat([left,right],dim=-1)).squeeze(-1)
