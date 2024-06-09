import torch

import numpy as np

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn

class DDPMSchedulerBase:

    def __init__(self, beta_min=0.0001, beta_max=0.02, n_steps=1000, mode='linear'):

        self.beta_min, self.beta_max, self.n_steps = beta_min, beta_max, n_steps

        if mode == 'linear':
            self.beta, self.alpha, self.alpha_hat = self._linear_scheduling()
        else: 
            raise NotImplementedError
        

    def _linear_scheduling(self):

        beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        alpha = 1. - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        
        return beta, alpha, alpha_hat
    

    def sample(self, model, bs, n_points=2048, nf=3, emb=None, save_process=False):
        """
            Args:
                - model        : neural net for noise prediction
                - bs           : number of samples to generate
                - n_points     : number of points per point cloud
                - nf           : number of features - default 3 for xyz coordinates
                - emb          : conditional embedding, if None it will be ignored
                - save_process : save the intermediate point clouds of the generation process
        """
        device = next(model.parameters()).device
        shape = (bs, n_points, nf)

        x_t = self.create_noise(shape, device)
        preds = [x_t.detach().cpu()] #.reshape(shape)

        for t in reversed(range(self.n_steps)):
            x_t = self.sample_step(model, x_t, t, emb, shape, device)
            if save_process: preds.append(x_t.detach().cpu()) #.reshape(shape)

        return preds if save_process else x_t.detach().cpu() #.reshape(shape)


    def sample_step(self, model, x_t, t, emb, shape, device):
        """
            Args:
                - model  : neural net for noise prediction
                - x_t    : previous point cloud
                - t      : current time step
                - emb    : conditional embedding, if None it will be ignored
                - shape  : shape of the point cloud
                - device : device to run the computations
        """
        bs = shape[0]

        # creating the time embedding variable
        t_batch = torch.full((bs,), t, device=device, dtype=torch.long)

        # activate the model to predict the noise
        noise_pred = model((x_t, t_batch)) if emb is None else model((x_t, t_batch, emb))
        
        # calculate the new point coordinates
        x_t = self.update_rule(x_t, noise_pred, t, bs, device)
        
        return x_t

    def create_noise(self, shape, device):
        return torch.randn(shape).to(device)
    
    def predict_x0_from_noise(self, x_t, noise_pred, t, bs, device):
        # x_t.shape : B x N x F
        # noise_pred.shape : B x N x F
        x0 =    

    def update_rule(self, x_t, noise_pred, t, bs, device):
        # x_t.shape : B x N x F
        # noise_pred.shape : B x N x F
        

    def noisify_sample(self, x0, step):
        raise NotImplementedError
    



class DDPMSparseScheduler(DDPMSchedulerBase):
    
    def __init__(self, beta_min=0.0001, beta_max=0.02, n_steps=1000, pres=1e-8, mode='linear'):
        super().__init__(beta_min, beta_max, n_steps, mode)
        self.pres = pres
    
    def sparse_from_pts(self, pts:torch.Tensor, shape):
        # Receive a tensor of points and return a SparseTensor 

        pts = pts.reshape(shape)
        # make coordinates positive
        coords = pts[:, :, :3]
        coords = coords - coords.min(dim=1, keepdim=True)[0]
        coords = coords.numpy()

        # unfortunately we need to loop over the batch for sparse_quantize
        batch = []
        for b in range(shape[0]):

            c, indices = sparse_quantize(coords[b], self.pres, return_index=True)
            f = pts[b][indices]

            batch.append(
                {'pc':SparseTensor(coords = torch.tensor(c), feats=f)}
            )
        
        batch = sparse_collate_fn(batch)['pc']

        return batch


    def update_rule(self, x_t, noise_pred, t, bs, device):
        a_t, abar_t, s_t = self.alpha[t], self.alpha_hat[t], self.sigma[t]
        
        new_pts = 1 / a_t.sqrt() * (x_t - (1 - a_t) / abar_t.sqrt() * noise_pred) + s_t * self.create_noise(x_t.shape, device)
        new_pts = new_pts.detach().cpu().reshape(bs, -1, self.nf)
        
        return self.noisify_sample(new_pts, t)



    def predict_x0_from_noise(self, x_t, noise_pred, t, bs, device):
        a_t, abar_t, s_t = self.alpha[t], self.alpha_hat[t], self.sigma[t]
        
        new_pts = 1 / a_t.sqrt() * (x_t - (1 - a_t) / abar_t.sqrt() * noise_pred) + s_t * self.create_noise(x_t.shape, device)
        new_pts = new_pts.detach().cpu().reshape(bs, -1, self.nf)
        
        return new_pts.to(device)