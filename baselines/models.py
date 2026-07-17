import torch
from torch import nn
from models.lstm_encoder import LSTMEncoder
from models.transformer_encoder import TransformerEncoder
from models.meta_gnn_rul import MetaGNNRUL

class EncoderRegressor(nn.Module):
    def __init__(self, encoder, dim=128):
        super().__init__(); self.encoder = encoder; self.head = nn.Linear(dim, 1)
    def forward(self, x): return self.head(self.encoder(x)[0]).squeeze(-1)

class CNNLSTM(nn.Module):
    def __init__(self, sensors, hidden=128, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(sensors, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, batch_first=True); self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2); _, (h, _) = self.lstm(z)
        return self.head(h[-1]).squeeze(-1)

def build_model(name, sensor_num, cfg):
    h=cfg["hidden_dim"]; e=cfg["embedding_dim"]; d=cfg["dropout"]
    if name in {"lstm", "reptile_lstm"}: return EncoderRegressor(LSTMEncoder(sensor_num,h,e,d),e)
    if name == "cnn_lstm": return CNNLSTM(sensor_num,h,d)
    if name == "transformer": return EncoderRegressor(TransformerEncoder(sensor_num,h,e,cfg["gat_heads"],d),e)
    if name in {"gnn", "meta_gnn", "no_attention", "no_gat"}:
        return MetaGNNRUL(sensor_num,h,e,cfg["gat_heads"],d,cfg["graph_k"],name!="no_gat",name!="no_attention",
                          cfg.get("graph_method","cosine"),cfg.get("dtw_downsample",5),cfg.get("self_attention_heads",4))
    raise ValueError(f"Unknown model: {name}")
