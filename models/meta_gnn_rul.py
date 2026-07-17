import torch
from torch import nn
from .lstm_encoder import LSTMEncoder
from .gat_encoder import GATEncoder
from .sensor_attention import TemporalSelfAttention
from .se_block import SensorSEBlock
from .rul_predictor import RULPredictor
from preprocess.graph_builder import build_knn_graph,build_dtw_graph

class MetaGNNRUL(nn.Module):
    def __init__(self, sensor_num, hidden_dim=128, embedding_dim=128, gat_heads=4, dropout=0.2,
                 graph_k=5, use_gat=True, use_sensor_attention=True, graph_method="cosine",dtw_downsample=5,
                 self_attention_heads=4):
        super().__init__()
        self.se_block=SensorSEBlock(sensor_num)
        self.temporal = LSTMEncoder(sensor_num, hidden_dim, embedding_dim, dropout)
        self.use_gat, self.use_sensor_attention, self.graph_k = use_gat, use_sensor_attention, graph_k
        self.graph_method,self.dtw_downsample=graph_method,dtw_downsample
        self.gat = GATEncoder(embedding_dim, gat_heads, dropout) if use_gat else nn.Identity()
        self.sensor_attention = TemporalSelfAttention(hidden_dim,embedding_dim,self_attention_heads,dropout) if use_sensor_attention else None
        fusion_dim = embedding_dim * (2 if use_sensor_attention else 1)
        self.predictor = RULPredictor(fusion_dim,dropout)
        self.pairwise_predictor=nn.Sequential(nn.Linear(fusion_dim*2,256),nn.LeakyReLU(0.2),nn.Dropout(dropout),nn.Linear(256,1))
    def forward(self, x, return_attention=False):
        recalibrated,se_weights=self.se_block(x)
        temporal, sequence = self.temporal(recalibrated)
        if self.use_gat:
            edge_index = build_dtw_graph(recalibrated,self.graph_k,self.dtw_downsample) if self.graph_method=="dtw" else build_knn_graph(temporal.detach(), self.graph_k)
            graph = self.gat(temporal, edge_index)
        else: graph = temporal
        attention = None
        if self.use_sensor_attention:
            sensor, attention = self.sensor_attention(sequence)
            graph = torch.cat([graph, sensor], dim=-1)
        pred = self.predictor(graph)
        return (pred,{"features":graph,"temporal_attention":attention,"sensor_weights":se_weights}) if return_attention else pred
