import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['lin', 'timestep_embedding', 'masked_softmax']

def lin(ni, nf, act=nn.SiLU, norm=None, bias=True):
    layers = nn.Sequential()
    if norm: layers.append(norm(ni))
    if act : layers.append(act())
    layers.append(nn.Linear(ni, nf, bias=bias))
    return layers

def timestep_embedding(tsteps, emb_dim, max_period= 10000):
    exponent = -math.log(max_period) * torch.linspace(0, 1, emb_dim//2, device=tsteps.device)
    emb = tsteps[:,None].float() * exponent.exp()[None,:]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    return F.pad(emb, (0,1,0,0)) if emb_dim%2==1 else emb


def masked_softmax(x, mask):
    # calculate softmax only for the masked elements
    # x.shape : B x H x N x N 
    # mask.shape : B x N
    
    mask = mask.unsqueeze(1) # broadcast across H dim
    mask = mask.unsqueeze(2) # broadcast across N dim
    # mask.shape B x 1 x 1 x N

    x_exp = x.exp() * mask
    x = x_exp / x_exp.sum(-1, keepdims=True)

    return x