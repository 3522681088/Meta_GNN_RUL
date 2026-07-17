import torch
from models.meta_gnn_rul import MetaGNNRUL

def test_forward_and_paper_aligned_dimensions():
    model=MetaGNNRUL(sensor_num=14,hidden_dim=32,embedding_dim=64,gat_heads=4,self_attention_heads=4)
    prediction,aux=model(torch.randn(6,50,14),return_attention=True)
    assert prediction.shape==(6,)
    assert aux["features"].shape==(6,128)
    assert aux["sensor_weights"].shape==(6,14)

