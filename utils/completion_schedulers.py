import torch
from torchsparse import SparseTensor
from utils.schedulers import SchedulerBase, SchedulingStrategy, DDPM
from abc import abstractmethod
from models.sparse_utils import batch_sparse_quantize_torch

class CompletionScheduler(SchedulerBase):

    @abstractmethod
    def get_pc(self, x_t, shape):
        pass

    @abstractmethod
    def pad_noise(x0, n_points):
        pass
    
    @torch.no_grad()
    def complete(self, x0, model, n_points, emb=None, save_process=False):
        # get the model device
        device = next(model.parameters()).device

        # take x_0 to device
        x0 = x0.to(device)

        shape = torch.Size((x0.shape[0], n_points, x0.shape[-1] + 1))
        
        # pad noise to point cloud
        x_t = self.pad_noise(x0, n_points)
        if save_process: preds = [self.get_pc(x_t, shape)] 
        
        for i, t in enumerate(self.strategy.steps):
            x_t = self.sample_step(model, x_t, t, i, emb, shape, device)
            if save_process: preds.append(self.get_pc(x_t, shape))
        
        return preds if save_process else self.get_pc(x_t, shape)

class SparseCompletionScheduler(CompletionScheduler):

    def __init__(self, strategy:SchedulingStrategy, save_process=False, pres=1e-5):
        super().__init__(strategy, save_process)
        self.pres = pres

    def pad_noise(self, x0, n_points):
        B, N, F = x0.shape
        num_pad = n_points - N
        noise = torch.randn(B, num_pad, F).to(x0.device)
        padded_pc = torch.cat([x0, noise], dim=1)

        mask = torch.zeros(B, n_points, 1).to(x0.device)
        mask[:, :N] = 1
        # print(N)
        # print(x0.shape)
        # print(n_points)
        # print(mask.sum())
        # print(mask.shape)

        padded_pc = torch.cat([padded_pc, mask], dim=-1)

        return self.torch2sparse(padded_pc, padded_pc.shape)       

    def update_rule(self, x_t, noise_pred, t, i, shape, device):
        mask = x_t.F[:, -1].bool()

        # print(mask.sum())
        
        x_t_new = self.strategy.update_rule(x_t.F[:, :3], noise_pred[:, :3], t, i, shape, device)
        
        x_t = x_t.F[:, :3]* mask.unsqueeze(-1) + x_t_new * ~mask.unsqueeze(-1)
        
        x_t = torch.cat([x_t, mask.unsqueeze(-1)], dim=-1)

        return self.torch2sparse(x_t, shape).to(device)

    def get_pc(self, x_t, shape):
        return x_t.F.detach().cpu().reshape(shape)[:, :, :3]

class SparseCompletionSchedulerGPU(SparseCompletionScheduler):

    def torch2sparse(self, pts:torch.Tensor, shape):
        pts = pts.reshape(shape)
        
        coords = pts[..., :3] # In case points have additional features
        coords = coords - coords.min(dim=1, keepdim=True).values
        coords, indices = batch_sparse_quantize_torch(coords, voxel_size=self.pres, return_index=True, return_batch_index=False)
        feats = pts.view(-1, shape[-1])[indices]

        return SparseTensor(coords=coords, feats=feats).to(coords.device)

class DDPMSparseCompletionSchedulerGPU(SparseCompletionSchedulerGPU):
    def __init__(self, beta_min=0.0001, beta_max=0.02, n_steps=1000, mode='linear', sigma='bt', pres=1e-5, save_process=False):
        strategy = DDPM(beta_min=beta_min, beta_max=beta_max, n_steps=n_steps, mode=mode, sigma=sigma)
        super().__init__(strategy, save_process=save_process, pres=pres)

