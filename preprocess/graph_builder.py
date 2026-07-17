import torch
import torch.nn.functional as F

def build_knn_graph(node_features, k=5, method="cosine"):
    """Build a directed kNN graph inside the current batch; returns edge_index [2,E]."""
    n = node_features.size(0)
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=node_features.device)
    k = min(k, n - 1)
    if method != "cosine": raise ValueError("Use build_dtw_graph for DTW")
    z = F.normalize(node_features, dim=-1)
    sim = z @ z.T
    sim.fill_diagonal_(-float("inf"))
    dst = sim.topk(k, dim=1).indices.reshape(-1)
    src = torch.arange(n, device=z.device).repeat_interleave(k)
    edges = torch.stack([src, dst])
    reverse = edges.flip(0)
    loops = torch.arange(n, device=z.device).repeat(2, 1)
    return torch.cat([edges, reverse, loops], dim=1)

def _dtw_distance(a,b):
    """DTW on 1-D degradation summaries; deliberately non-differentiable graph construction."""
    n,m=a.numel(),b.numel(); prev=torch.full((m+1,),float("inf"),device=a.device); prev[0]=0.0
    for i in range(1,n+1):
        cur=torch.full((m+1,),float("inf"),device=a.device)
        for j in range(1,m+1): cur[j]=torch.abs(a[i-1]-b[j-1])+torch.minimum(prev[j],torch.minimum(cur[j-1],prev[j-1]))
        prev=cur
    return prev[m]

def build_dtw_graph(windows,k=5,downsample=5):
    n=windows.size(0)
    if n<=1: return torch.zeros((2,0),dtype=torch.long,device=windows.device)
    k=min(k,n-1); trajectories=windows[:,::downsample,:].mean(dim=-1).detach()
    distance=torch.zeros((n,n),device=windows.device)
    for i in range(n):
        for j in range(i+1,n):
            d=_dtw_distance(trajectories[i],trajectories[j]); distance[i,j]=d; distance[j,i]=d
    distance.fill_diagonal_(float("inf")); dst=distance.topk(k,dim=1,largest=False).indices.reshape(-1)
    src=torch.arange(n,device=windows.device).repeat_interleave(k); edges=torch.stack([src,dst])
    loops=torch.arange(n,device=windows.device).repeat(2,1)
    return torch.cat([edges,edges.flip(0),loops],dim=1)
