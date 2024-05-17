import torch
import os

from pclab.learner import Callback

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