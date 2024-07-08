import torch
import os
import torchsparse
import fastcore.all as fc
from typing import Mapping
from copy import copy

from pclab.learner import Callback, to_cpu
from pclab.utils import def_device

from torcheval.metrics import Mean

__all__ = ['DDPMCB', 'CheckpointCB', 'GradientClipCB', 'DeviceCBSparse', 'LossCB']

class DDPMCB(Callback):
    '''
    This is the base DDPC callback, used for the generation task. 
    Other tasks like super-resolution and completion use a different version.
    '''
    def before_batch(self, learn): 
        pts = learn.batch['input']
        t = torch.tensor(learn.batch['t'])
        noise = learn.batch['noise']
        inp = (pts, t)
        learn.batch = (inp, noise.F)



class CheckpointCB(Callback):
    def __init__(self, epoch_step, run_name, path='./checkpoints/', run_params={}, replace_checkpoints=False):
        self.epoch_step = epoch_step
        self.path = path
        self.replace_checkpoints = replace_checkpoints
        self.run_name, self.run_params = run_name, run_params
    
    def after_epoch(self, learn):
        if (learn.epoch + 1) % self.epoch_step == 0: # do not save a checkpoint at epoch 0
            self.run_params['c_epoch'] = learn.epoch + 1
            subfix = '.pt' if self.replace_checkpoints else f'_{learn.epoch+1}.pt'
            path = os.path.join(self.path, self.run_name + subfix) 
            
            torch.save({
                'run_params':self.run_params, 
                'state_dict':learn.model.state_dict()
            }, path)
    
    def after_fit(self, learn):
        # save a checkpoint after training is completed
        self.run_params['c_epoch'] = learn.epoch + 1
        subfix = '.pt'
        path = os.path.join(self.path, self.run_name + subfix) 
            
        torch.save({
            'run_params':self.run_params, 
            'state_dict':learn.model.state_dict()
        }, path)

class GradientClipCB(Callback):
    def __init__(self, clip_value=10.0):
        self.clip_value = clip_value
    
    def after_backward(self, learn):
        torch.nn.utils.clip_grad_norm_(learn.model.parameters(), self.clip_value)



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