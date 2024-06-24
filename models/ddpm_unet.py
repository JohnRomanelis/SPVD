import torch
import torch.nn as nn
import torch.nn.functional as F

import torchsparse
import torchsparse.nn as spnn
from .sparse_utils import PointTensor, initial_voxelize, voxel_to_point, point_to_voxel
from .utils import lin, timestep_embedding

import torch_geometric as tg
from torch_geometric.nn import MessagePassing

import fastcore.all as fc
from functools import wraps

class TimeEmbeddingBlock(MessagePassing):
    """ A block to incorporate the time embedding information to a sparse representation of node features. 
        Use to intergrade the time embedding information to sparse voxels
    """

    def __init__(self, n_emb, embed_dim):
        super().__init__('add', flow='target_to_source')

        self.proj_mlp = nn.Linear(n_emb, embed_dim * 2)

    def forward(self, x, t, b):
        # x: features of each node
        # t: time embedding for each point cloud
        # b: a tensor indicate the batch index of each node

        node_idx = torch.arange(0, len(x), device=x.device)
        edge_index = torch.stack([node_idx, b]).long()

        #print(edge_index.shape)
        
        t = self.proj_mlp(F.silu(t)) #[:, :]
        #print(t.shape)

        return  self.propagate(edge_index=edge_index, x=x, t=t)
    
    
    def message(self, x_i, t_j):

        scale, shift = torch.chunk(t_j, 2, dim=-1)

        out = (1 + scale) * x_i + shift

        return out
    

def saved(m, blk):
    m_ = m.forward

    @wraps(m.forward)
    def _f(*args, **kwargs):
        res = m_(*args, **kwargs)
        blk.saved.append(res)
        return res

    m.forward = _f
    return m

def unet_conv(ni, nf, ks=2, bias=True):
    layers = nn.Sequential(
        spnn.BatchNorm(ni),
        spnn.SiLU(), 
        spnn.Conv3d(ni, nf, ks, bias=bias)
    )
    return layers

class EmbResBlock(nn.Module):
    def __init__(self, n_emb, ni, nf=None, ks=3):
        super().__init__()
        nf = nf or ni

        self.conv1 = unet_conv(ni, nf, ks)
        self.conv2 = unet_conv(nf, nf, ks)

        self.t_emb = TimeEmbeddingBlock(n_emb, nf)
        
        self.idconv = fc.noop if ni==nf else nn.Linear(ni, nf) 

    
    def forward(self, x_in, t):

        # first conv
        x = self.conv1(x_in)

        # time embedding
        x.F = self.t_emb(x.F, t, x.C[:, 0])

        # second conv
        x = self.conv2(x)

        # residual connection
        x.F = x.F + self.idconv(x_in.F)

        return x
    
class DownBlock(nn.Module):
    def __init__(self, n_emb, ni, nf, add_down=True, num_layers=1):
        super().__init__()
        self.resnets = nn.ModuleList([saved(EmbResBlock(n_emb, ni if i==0 else nf, nf), self)
                                      for i in range(num_layers)])
        
        self.down = saved(spnn.Conv3d(nf, nf, 2, stride=2),self) if add_down else nn.Identity()
            
    def forward(self, x, t):
        self.saved = []
        for resnet in self.resnets: x = resnet(x, t)
        x = self.down(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, n_emb, ni, prev_nf, nf, add_up=True, num_layers=2):
        super().__init__()
        self.resnets = nn.ModuleList(
            [EmbResBlock(n_emb, (prev_nf if i==0 else nf)+(ni if (i==num_layers-1) else nf), nf)
            for i in range(num_layers)])
        
        self.up = spnn.Conv3d(nf, nf, 2, stride=2, transposed=True) if add_up else nn.Identity()

    def forward(self, x, t, ups):
        for resnet in self.resnets: x = resnet(torchsparse.cat([x, ups.pop()]), t)
        return self.up(x)
    
class SPVUnet(nn.Module):

    def __init__(self, voxel_size, in_channels=3, out_channels=3, nfs=(224,448,672,896), pres=1e-8, num_layers=1):
        super().__init__()

        self.pres = pres
        self.voxel_size = voxel_size
        
        # This is crucial in the sparse implementation to correct dublicate points etc 
        # before the introduction of skip connections
        self.conv_in = spnn.Conv3d(in_channels, nfs[0], kernel_size=3, padding=1)
        
        
        self.n_temb = nf = nfs[0]
        n_emb = nf*4
        self.emb_mlp = nn.Sequential(lin(self.n_temb, n_emb, norm=nn.BatchNorm1d),
                                     lin(n_emb, n_emb))
        
        
        self.downs = nn.ModuleList()
        for i in range(len(nfs)):
            ni = nf
            nf = nfs[i]
            self.downs.append(DownBlock(n_emb, ni, nf, add_down=i!=len(nfs)-1, num_layers=num_layers))
            
        self.mid_block = EmbResBlock(n_emb, nfs[-1])

        rev_nfs = list(reversed(nfs))
        nf = rev_nfs[0]
        self.ups = nn.ModuleList()
        for i in range(len(nfs)):
            prev_nf = nf
            nf = rev_nfs[i]
            ni = rev_nfs[min(i+1, len(nfs)-1)]
            self.ups.append(UpBlock(n_emb, ni, prev_nf, nf, add_up=i!=len(nfs)-1, num_layers=num_layers+1))
             
        self.conv_out = nn.Sequential(
            nn.BatchNorm1d(nfs[0]),
            nn.SiLU(), 
            nn.Linear(nfs[0], nfs[0], bias=False), # extra
            nn.BatchNorm1d(nfs[0]),                # extra
            nn.SiLU(),                             # extra
            nn.Linear(nfs[0], 3, bias=False),
        )
        
        
        
    def forward(self, inp):

        # Input Processing
        x, t = inp
        z = PointTensor(x.F, x.C.float())
        t = timestep_embedding(t, self.n_temb)
        emb = self.emb_mlp(t)
        
        
        # map the point tensor to the sparse tensor
        x0 = initial_voxelize(z, self.pres, self.voxel_size)
        # pass through an initial convolution to extract local features
        x = self.conv_in(x0)
        
        saved = [x]

        for block in self.downs:
            x = block(x, emb)
        saved += [p for o in self.downs for p in o.saved]
                
        x = self.mid_block(x, emb)
        
        for block in self.ups:
            x = block(x, emb, saved)
        
        z1 = voxel_to_point(x, z)
        
        return self.conv_out(z1.F)
    
