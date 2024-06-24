# imports
import torch
import math
from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from models.sparse_utils import batch_sparse_quantize_torch

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

    def get_pc(self, x_t, shape):
        # this functions receives x_t as used by the pipeline returns a cpu tensor
        return x_t.detach().cpu().reshape(shape)
        
    @torch.no_grad()
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
        preds = [self.get_pc(x_t, shape)] 

        for t in reversed(range(self.n_steps)):
            x_t = self.sample_step(model, x_t, t, emb, shape, device)
            if save_process: preds.append(self.get_pc(x_t, shape)) 

        return preds if save_process else self.get_pc(x_t, shape)


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
        x_t = self.update_rule(x_t, noise_pred, t, shape, device)
        
        return x_t

    def create_noise(self, shape, device):
        return torch.randn(shape).to(device)

    def predict_x0_from_noise(self, x_t, noise_pred, t, shape, device):
        # x_t.shape : B x N x F
        # noise_pred.shape : B x N x F
        raise NotImplementedError

    def update_rule(self, x_t, noise_pred, t, shape, device):
        # x_t.shape : B x N x F
        # noise_pred.shape : B x N x F
        raise NotImplementedError

    def noisify_sample(self, x0, step):
        raise NotImplementedError

class DDPMSparseScheduler(DDPMSchedulerBase):

    def __init__(self, beta_min=0.0001, beta_max=0.02, n_steps=1000, pres=1e-5, mode='linear', sigma='bt'):
        super().__init__(beta_min, beta_max, n_steps, mode)
        self.pres = pres

        assert sigma in ['bt', 'coef_bt'], sigma
        if sigma == 'bt':
            self.sigma = self.beta.sqrt()
        else:
            alpha_hat_prev1 = torch.ones_like(self.alpha_hat)
            alpha_hat_prev1[1:] = self.alpha_hat[:-1]
            self.sigma = torch.sqrt(self.beta * (1 - alpha_hat_prev1) / (1 - self.alpha_hat))
        
    def torch2sparse(self, pts:torch.Tensor, shape):
        # Receive a torch.Tensor of shape BxNxF and returns a SparseTensor representation
        pts = pts.cpu().reshape(shape) # make sure points have the correct shape
        
        # make coordinates positive
        coords = pts[:, :, :3]
        coords = coords - coords.min(dim=1, keepdim=True)[0]
        coords = coords.numpy()

        # Unfortunately we need to loop over the batch to apply sparse_quantize 
        # Also DATA have to be in CPU and coords represented as np.arrays
        batch = []
        for b in range(shape[0]):

            c, indices = sparse_quantize(coords[b], self.pres, return_index=True)
            f = pts[b][indices]

            batch.append(
                {'pc':SparseTensor(coords = torch.tensor(c), feats=f)}
            )
        
        batch = sparse_collate_fn(batch)['pc']

        return batch

    def create_noise(self, shape, device):
        noise = torch.randn(shape)
        noise = self.torch2sparse(noise, shape).to(device)
        return noise

    def update_rule(self, x_t, noise_pred, t, shape, device):

        # outside the update_rule the point cloud is represented as SparseTensor
        x_t = x_t.F

        # create normal noise with the same shape as x_t
        z = torch.randn(x_t.shape).to(device)
        
        # get parameters for the current timestep
        a_t, ahat_t, s_t = self.alpha[t], self.alpha_hat[t], self.sigma[t]
        
        x_t = 1 / math.sqrt(a_t) * (x_t - (1 - a_t) / (math.sqrt(1 - ahat_t)) * noise_pred) + s_t * z

        # so we should turn it back to a sparse tensor before return
        return self.torch2sparse(x_t, shape).to(device)

    def get_pc(self, x_t, shape):
        # this functions receives x_t as used by the pipeline returns a cpu tensor
        return x_t.F.detach().cpu().reshape(shape)


class DDPMSparseSchedulerGPU(DDPMSparseScheduler):

    def torch2sparse(self, pts:torch.Tensor, shape):
        pts = pts.reshape(shape)
        
        coords = pts[..., :3] # In case points have additional features
        coords = coords - coords.min(dim=1, keepdim=True).values
        coords, indices = batch_sparse_quantize_torch(coords, voxel_size=self.pres, return_index=True, return_batch_index=False)
        feats = pts.view(-1, 3)[indices]

        return SparseTensor(coords=coords, feats=feats).to(coords.device)

