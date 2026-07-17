import torch
from torch import nn

class SensorSEBlock(nn.Module):
    """Squeeze-and-Excitation over sensor channels, adapted from MetaFluAD's SEBlock."""
    def __init__(self, sensor_num, reduction=4):
        super().__init__()
        hidden=max(4,sensor_num//reduction)
        self.excitation=nn.Sequential(nn.Linear(sensor_num,hidden),nn.ReLU(),nn.Linear(hidden,sensor_num),nn.Sigmoid())
    def forward(self,x):
        weights=self.excitation(x.mean(dim=1))
        return x*weights.unsqueeze(1),weights

