from models.ddpm_unet_attn import SPVUnet

import torch
import torch.nn as nn
from datasets.shapenet_pointflow_sparse import ShapeNet15kPointCloudsSparseNoisy
from functools import partial
from torch.utils.data import DataLoader
import torchsparse
from torchsparse.utils.collate import sparse_collate_fn
from pclab.utils import DataLoaders

path = "/home/vvrbeast/Desktop/Giannis/Data/ShapeNetCore.v2.PC15k"

tr_dataset = ShapeNet15kPointCloudsSparseNoisy(
            categories= ['car'], split='train',
            tr_sample_size=2048,
            te_sample_size=2048,
            scale=1.,
            root_dir=path,
            normalize_per_shape=False,
            normalize_std_per_axis=False, 
            random_subsample=True)

te_dataset = ShapeNet15kPointCloudsSparseNoisy(
            categories= ['car'], split='val',
            tr_sample_size=2048,
            te_sample_size=2048,
            scale=1.,
            root_dir=path,
            normalize_per_shape=False,
            normalize_std_per_axis=False,
            random_subsample=True)

tr_dataset.set_voxel_size(1e-5)
tr_dataset.set_noise_params(beta_max=0.02)

te_dataset.set_voxel_size(1e-5)
te_dataset.set_noise_params(beta_max=0.02)

train_dl, valid_dl = map(partial(DataLoader, batch_size=32, shuffle=True, num_workers=8, drop_last=True, collate_fn=sparse_collate_fn), (tr_dataset, te_dataset))
dls = DataLoaders(train_dl, valid_dl)


from pclab.learner import *
from pclab.utils import def_device
import fastcore.all as fc
from typing import Mapping
from copy import copy
from torcheval.metrics import Mean
from utils.callbacks import GradientClipCB
from utils.callbacks import CheckpointCB
from models.sparse_utils import PointTensor

class DDPMCB(Callback):
    
    def before_batch(self, learn): 
        pts = learn.batch['input']
        t = torch.tensor(learn.batch['t'])
        noise = learn.batch['noise']
        inp = (pts, t)
        learn.batch = (inp, noise.F)

def to_device(x, device=def_device):
    if isinstance(x, (torch.Tensor, torchsparse.SparseTensor)): return x.to(device)
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)

class DeviceCBSparse(Callback):
    order = DDPMCB.order + 1
    def __init__(self, device=def_device): fc.store_attr()
    def before_fit(self, learn):
        if hasattr(learn.model, 'to'): learn.model.to(self.device)
    def before_batch(self, learn): learn.batch = to_device(learn.batch, device=self.device)


class LossCB(Callback):
    def __init__(self, *ms, **metrics):
        for o in ms: metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, d): print(d)
    def before_fit(self, learn): learn.metrics = self
    def before_epoch(self, learn): [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, learn):
        log = {k:f'{v.compute():.3f}' for k,v in self.all_metrics.items()}
        log['epoch'] = learn.epoch
        log['train'] = 'train' if learn.model.training else 'eval'
        self._log(log)

    def after_batch(self, learn):
        x,y,*_ = learn.batch
        #for m in self.metrics.values(): m.update(to_cpu(learn.preds), y)
        self.loss.update(to_cpu(learn.loss), weight=2)

lr = 0.0001 
epochs = 2000

model = SPVUnet()

# scheduler
total_steps = epochs * len(dls.train)
sched = partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=lr, total_steps = total_steps)

# Callbacks
ddpm_cb = DDPMCB()
checkpoint_cb = CheckpointCB(1000, 'originalEmbResBlock_large_car', run_params={'msg':model.msg})
cbs = [ddpm_cb, DeviceCBSparse(), ProgressCB(plot=False), LossCB(), GradientClipCB(), checkpoint_cb, BatchSchedCB(sched)]

learn = TrainLearner(model, dls, nn.MSELoss(), lr=lr, cbs=cbs, opt_func=torch.optim.Adam)
learn.fit(epochs)