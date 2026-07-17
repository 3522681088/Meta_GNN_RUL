from torch import nn

class TemporalSelfAttention(nn.Module):
    """Parallel multi-head self-attention branch corresponding to Eq. (6) of MetaFluAD."""
    def __init__(self, input_dim, embedding_dim=256, heads=4, dropout=0.2):
        super().__init__()
        self.input_proj=nn.Linear(input_dim,embedding_dim)
        self.attention=nn.MultiheadAttention(embedding_dim,heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(embedding_dim)
        self.ffn=nn.Sequential(nn.Linear(embedding_dim,embedding_dim*4),nn.ReLU(),nn.Dropout(dropout),nn.Linear(embedding_dim*4,embedding_dim))
        self.norm2=nn.LayerNorm(embedding_dim)
    def forward(self,sequence):
        z=self.input_proj(sequence)
        attended,weights=self.attention(z,z,z,need_weights=True,average_attn_weights=False)
        z=self.norm1(z+attended); z=self.norm2(z+self.ffn(z))
        return z.mean(dim=1),weights
