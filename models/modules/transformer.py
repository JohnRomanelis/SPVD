import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch_geometric.utils import to_dense_batch

from ..utils import masked_softmax

__all__ = ['SparseAttention', 'SparseTransformer']
           

def sparse_to_dense(x, b):
    "Receives a sparse representation and the batch idx for each element and returns a dense sequence representation and a mask indicating the actual values"
    feats_dense, mask = to_dense_batch(x, batch=b)
    return feats_dense, mask

class Mlp(nn.Module):
    """
        MLP module for the transformer feedforward block.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    """
        Attention module that can handle a masked input.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k , v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        #attn = attn.softmax(dim=-1)
        attn = masked_softmax(attn, mask)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class TransformerBlock(nn.Module):
    """
        Transformer Block that can handle a masked input.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)

        # NOTE: Should test if this is better than dropout
        self.drop_path = DropPath(drop_path) if drop_path >0. else nn.Identity()

        # ATTENTION BLOCK
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # MLP BLOCK
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x * mask.unsqueeze(-1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))    
        x = x * mask.unsqueeze(-1)
        return x
    
# -------------------------------------------------------- #

class SparseAttention(nn.Module):
    """
        An attention module that works with sparse tensors.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)

        # ATTENTION BLOCK
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        #self.emb_proj = nn.Linear(dim, 2*dim)

    def forward(self, x):
            
        x_dense, mask = sparse_to_dense(x.F, x.C[:, 0].long())
            
        x_dense = x_dense + self.attn(self.norm1(x_dense), mask)
        x.F = x_dense[mask]

        return x
    
class SparseTransformer(nn.Module):
    """
        A series of transformer blocks that works with sparse tensors.
    """
    def __init__(self, depth, dim, num_heads, drop_path=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, drop_path=drop_path) for _ in range(depth)
        ])

    
    def forward(self, x, emb):

        x_dense, mask = sparse_to_dense(x.F, x.C[:, 0].long())
       
        for b in self.blocks:
            x_dense = b(x_dense, mask)

        x.F = x_dense[mask]

        return x