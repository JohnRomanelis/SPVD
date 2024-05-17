import numpy as np
import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn
import math

def sparse_noise_batch(bs, voxel_size=1e-8, n_points=2048):
    batch = []
    for i in range(bs):
        pts = np.random.normal(size=(n_points, 3))
        coords = pts - np.min(pts, axis=0, keepdims=True)

        coords, indices = sparse_quantize(coords, voxel_size, return_index=True)
        
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(pts[indices], dtype=torch.float)
        
        pc_sparse = SparseTensor(coords=coords, feats=feats)
        
        batch.append({'pc':pc_sparse})
        
    batch = sparse_collate_fn(batch)
    
    return batch['pc']

def sparse_from_pts(pts, bs, voxel_size=1e-8):

    pts = pts.reshape(bs, -1, 3)

    # make coordinates possitive
    coords = pts - pts.min(dim=1, keepdims=True).values
    coords = coords.numpy()
    
    batch = []
    for b in range(bs):

        c, index = sparse_quantize(coords[b], voxel_size, return_index=True)
        f = pts[b][index]
        
        batch.append(
            {'pc':SparseTensor(coords = torch.tensor(c), feats=f)}
        )

    batch = sparse_collate_fn(batch)['pc']

    return batch

@torch.no_grad()
def genSparseDDPM(model, bs, alpha, alphabar, sigma, n_steps=1000, n_points=2048):
    device = next(model.parameters()).device
    
    x_t = sparse_noise_batch(bs, n_points=n_points).to(device)

    for t in reversed(range(n_steps)):

        # creating the time embedding variable
        t_batch = torch.full((bs,), t, device=device, dtype=torch.long)

        # create random noise 
        z = (torch.randn(x_t.F.shape) if t>0 else torch.zeros(x_t.F.shape)).to(device)

        # activate the model to predict the noise
        noise_pred = model((x_t, t_batch))

        # calculate the new point coordinates
        pts = x_t.F # previous point coordinates in the continuous space

        a_t, abar_t, s_t = alpha[t], alphabar[t], sigma[t]
        
        new_pts = 1 / math.sqrt(a_t) * (pts - (1 - a_t) / (math.sqrt(1 - abar_t)) * noise_pred) + s_t * z

        # create a sparse tensor for the next step
        new_pts = new_pts.detach().cpu().reshape(bs, -1, 3)
        
        x_t = sparse_from_pts(new_pts, bs).to(device)
        
    return new_pts.to(device)


@torch.no_grad()
def denoiseSparseDDPM(model, x0, start_step, alpha, alphabar, sigma, n_steps=1000, n_points=2048):
    device = next(model.parameters()).device
    
    x_t = x0.to(device)

    bs = x_t.C[:,0].max() + 1

    for t in reversed(range(start_step, n_steps)):

        # creating the time embedding variable
        t_batch = torch.full((bs,), t, device=device, dtype=torch.long)

        # create random noise 
        z = (torch.randn(x_t.F.shape) if t>0 else torch.zeros(x_t.F.shape)).to(device)

        # activate the model to predict the noise
        noise_pred = model((x_t, t_batch))

        # calculate the new point coordinates
        pts = x_t.F # previous point coordinates in the continuous space

        a_t, abar_t, s_t = alpha[t], alphabar[t], sigma[t]
        
        new_pts = 1 / math.sqrt(a_t) * (pts - (1 - a_t) / (math.sqrt(1 - abar_t)) * noise_pred) + s_t * z

        # create a sparse tensor for the next step
        new_pts = new_pts.detach().cpu().reshape(bs, -1, 3)
        
        x_t = sparse_from_pts(new_pts, bs).to(device)
        
    return new_pts.to(device)

class DDPMScheduler:

    def __init__(self, beta_min, beta_max, n_steps):
        self.n_steps, self.beta_min, self.beta_max = n_steps, beta_min, beta_max
        
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        self.α = 1. - self.beta 
        self.ᾱ = torch.cumprod(self.α, dim=0)
        self.σ = self.beta.sqrt()
        
        self.beta = self.beta.numpy()
        self.α = self.α.numpy()
        self.ᾱ = self.ᾱ.numpy() 
        self.σ = self.σ.numpy()

    def sample(self, model, bs, n_points=2048):
        return genSparseDDPM(model, bs, self.α, self.ᾱ, self.σ, self.n_steps, n_points=n_points)
    
    def denoise_sample(self, model, x0, step):
        return denoiseSparseDDPM(model, x0, step, self.α, self.ᾱ, self.σ, self.n_steps)
    
    def noisify_sample(self, x0, step):

        noise = torch.randn(x0.shape)
        abar_t = self.ᾱ[step]
        xt = np.sqrt(abar_t)*x0 + np.sqrt(1-abar_t)*noise

        return xt

@torch.no_grad()
def genSparseDDIM(model, bs, step_inds, alphabar, n_points=2048):
    device = next(model.parameters()).device
    
    s_steps = len(step_inds)
    x_t = sparse_noise_batch(bs, n_points=n_points).to(device)

    for i in reversed(range(1, s_steps)):

        t = step_inds[i]
        # creating the time embedding variable
        t_batch = torch.full((bs,), t, device=device, dtype=torch.long)

        # predict the noise
        noise_pred = model((x_t, t_batch))

        # calculation of new point coordinates
        pts = x_t.F
        abar_t, abar_t1 = alphabar[i], alphabar[i-1]
        new_pts = math.sqrt(abar_t1) * ((pts - math.sqrt(1 - abar_t) * noise_pred) / math.sqrt(abar_t)) + (math.sqrt(1 - abar_t1) * noise_pred)

        # create a sparse tensor for the next step
        new_pts = new_pts.detach().cpu().reshape(bs, -1, 3)

        x_t = sparse_from_pts(new_pts, bs).to(device)

    return new_pts.to(device)

class DDIMScheduler(DDPMScheduler):

    def __init__(self, beta_min, beta_max, n_steps, s_steps):
        super().__init__(beta_min, beta_max, n_steps)
        assert s_steps < n_steps
        self.set_sampling_steps(s_steps)

    def set_sampling_steps(self, s_steps):
        self.s_steps = s_steps
        self.selected_inds = torch.floor(torch.linspace(0, 999, s_steps)).long()
        self.selected_alphabar = self.ᾱ[self.selected_inds]

    def sample(self, model, bs, n_points=2048):
        return genSparseDDIM(model, bs, self.selected_inds, self.selected_alphabar, n_points)
    
