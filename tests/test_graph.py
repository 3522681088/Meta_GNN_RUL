import torch
from preprocess.graph_builder import build_knn_graph,build_dtw_graph

def test_graph_shapes():
    z=torch.randn(8,16); windows=torch.randn(8,20,5)
    for edges in [build_knn_graph(z,k=2),build_dtw_graph(windows,k=2,downsample=4)]:
        assert edges.shape[0]==2
        assert edges.min()>=0 and edges.max()<8

