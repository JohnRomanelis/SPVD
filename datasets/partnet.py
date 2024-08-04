import h5py
import glob
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from datasets.utils import NoiseSchedulerDDPM
import random
from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from torch.utils.data import DataLoader
from torchsparse.utils.collate import sparse_collate_fn

categories = ['Faucet', 'Chair', 'Display', 'Knife', 'Table', 'Laptop', 'Refrigerator', 'Microwave', 
              'StorageFurniture', 'Bowl', 'Scissors', 'Door', 'TrashCan','Bed', 'Keyboard', 'Clock', 
              'Bottle', 'Bag', 'Lamp', 'Earphone', 'Vase', 'Dishwasher', 'Mug', 'Hat']

class PartNet(Dataset):
    """
        This dataset class loads data from .h5 and .json files and returns them unprocessed. 
    """
    def __init__(self, root_path, category, split='train', n_points=10000):
        super().__init__()

        all_categories = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
        assert category in all_categories, all_categories
        assert split in ['train', 'test', 'val'], split

        self.category = category
        self.split = split
        # create the path for the specific category
        category_path = os.path.join(root_path, category)
        
        # read all .h5 files and .json files
        self.points, self.labels, self.colors, self.meta = self.load_files(category_path, split)
        self.points = self.points[:, :n_points]
        self.labels = self.labels[:, :n_points]
        self.colors = self.colors[:, :n_points]

    def load_files(self, category_path, split):
        all_points, all_labels, all_colors = [], [], []
        all_meta = []
        file_pattern = f'{split}*.h5'
        file_paths = glob.glob(os.path.join(category_path, file_pattern))
        
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as f:
                points = f['pts'][:]
                labels = f['label'][:]
                colors = f['rgb'][:]
                all_points.append(points)
                all_labels.append(labels)
                all_colors.append(colors)

            json_file = file_path.split('.')[0] + '.json'
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_meta.extend(data)

        all_points = np.concatenate(all_points, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        
        return all_points, all_labels, all_colors, all_meta

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):

        pc, label, color, metadata = self.points[idx], self.labels[idx], self.colors[idx], self.meta[idx]
        return pc, label, color, metadata

class PartNetCompletion(Dataset):

    def __init__(self, root_path, category, split='train', n_points=2048):
        super().__init__()

        assert split in ['train', 'test', 'val'], split

        # path to load points and labels
        point_path = os.path.join(root_path, f'{category}_{split}_points.npy')
        label_path = os.path.join(root_path, f'{category}_{split}_labels.npy')

        # load points and labels
        points = np.load(point_path)
        labels = np.load(label_path)

        # keep n_points
        self.points = points[:, :n_points]
        self.labels = labels[:, :n_points]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pc, labels = self.points[idx], self.labels[idx]  
        return pc, labels

class PartNetCompletionNoisySparse(PartNetCompletion):

    def __init__(self, root_path, category, split='train', n_points=2048, pres=1e-5, min_num_parts=3, add_real_info=True,
                  ddpm_params = {'beta_min':0.0001, 
                                 'beta_max':0.02, 
                                 'n_steps' :1000, 
                                 'mode'    :'linear'}):
        super().__init__(root_path, category, split, n_points)
        self.pres = pres
        self.noise_scheduler = NoiseSchedulerDDPM(beta_min=ddpm_params['beta_min'],
                                                  beta_max=ddpm_params['beta_max'],
                                                  n_steps =ddpm_params['n_steps'],
                                                  mode    =ddpm_params['mode'])

        self.min_num_parts = min_num_parts
        self.add_real_info = add_real_info
        
    def __getitem__(self, idx):
        points, labels = super().__getitem__(idx)

        # normalize point cloud (mean=0, std=1)
        points = points - points.mean(axis=0)
        points = points / points.std()
        
        # select parts to discard
        unique_labels = np.unique(labels)
        num_parts = len(unique_labels)
        if num_parts > self.min_num_parts:
            keep_parts = random.randint(self.min_num_parts, num_parts - 1)
        else:
            keep_parts = random.randint(1, num_parts-1)
            
        keep_labels = np.random.choice(unique_labels, keep_parts, replace=False)
        
        # create a mask        
        labels = torch.tensor(labels)
        keep_labels = torch.tensor(keep_labels)
        mask = torch.isin(labels, keep_labels)

        # add noise
        points = torch.tensor(points)
        noisy_pc, t, noise = self.noise_scheduler(points)

        # keep the original points for the masked areas
        noisy_pc[mask] = points[mask]
                
        # voxelize
        coords = noisy_pc.numpy()
        coords = coords - np.min(coords, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords, self.pres, return_index=True)

        coords = torch.tensor(coords)

        if self.add_real_info:
            noisy_pc = torch.cat([noisy_pc, mask.unsqueeze(-1).float()], dim=-1)
        
        feats = noisy_pc[indices]
        noise = noise[indices]
        mask  = mask[indices]
        
        noisy_pc = SparseTensor(coords=coords, feats=feats)
        
        return {
            'input':noisy_pc,
            't': t,
            'noise': noise,
            'mask': mask
        }

def get_sparse_completion_datasets(path, category, n_points=2048, pres=1e-5, min_num_parts=3, add_real_info=True,
                                    ddpm_params = {'beta_min':0.0001, 
                                                   'beta_max':0.02, 
                                                   'n_steps':1000, 
                                                   'mode':'linear' }):

    tr_dataset = PartNetCompletionNoisySparse(path, category, split='train', n_points=n_points, pres=pres, min_num_parts=min_num_parts, 
                                                  add_real_info=add_real_info, ddpm_params=ddpm_params) 
    te_dataset = PartNetCompletionNoisySparse(path, category, split='test', n_points=n_points, pres=pres, min_num_parts=min_num_parts, 
                                                  add_real_info=add_real_info, ddpm_params=ddpm_params)
    
    return tr_dataset, te_dataset

def get_sparse_completion_dataloaders(path, category, n_points=2048, pres=1e-5, min_num_parts=3, add_real_info=True, batch_size=32, num_workers=8,
                                       ddpm_params = {'beta_min':0.0001, 
                                                      'beta_max':0.02, 
                                                      'n_steps':1000, 
                                                      'mode':'linear'}):
    tr_dataset, te_dataset = get_sparse_completion_datasets(path, category=category, n_points=n_points, pres=pres, min_num_parts=min_num_parts, 
                                                            add_real_info=add_real_info, ddpm_params=ddpm_params)

    tr_dl = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=sparse_collate_fn)
    te_dl = DataLoader(te_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=sparse_collate_fn)

    return tr_dl, te_dl

class ShapeNetColor(Dataset):
    def __init__(self, root_path, category, split='train', n_points=2048, max_points=10000):
        '''
        Processing: 
            1. Normalize color values globally (dataset provides inverse function to retrieve correct color values)
            2. Normalize point coordinates per shape ==> mean=0, std=1
        '''
        super().__init__()

        self.category, self.split, self.n_points, self.max_points = category, split, n_points, max_points
        
        data_path = os.path.join(root_path, f'{category}_{split}.npy')
        data = np.load(data_path)
        data = data[:, :max_points, :]

        # normalize color values
        self.m = data[..., 3:].mean(axis=1).mean(axis=0)
        data[..., 3:] = data[..., 3:] - self.m
        self.s = data[..., 3:].std()
        data[..., 3:] = data[..., 3:] / self.s

        self.data = data

    def get_actual_color(self, clr):
        return (clr * self.s + self.m).clamp(0, 1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.max_points > self.n_points:
            # select a random sabsample of the points
            indices = np.arange(self.n_points)
            np.random.shuffle(indices)
            data = data[indices]

        # normalize coordinates
        data[:, :3] = data[:, :3] - data[:, :3].mean(axis=0)
        data[:, :3] = data[:, :3] / data[:, :3].std()

        return torch.tensor(data)


class ShapeNetColorSparseNoisy(ShapeNetColor):

    def __init__(self, root_path, category, split='train', n_points=2048, max_points=10000, 
                 pres=1e-5, ddpm_params = {'beta_min':0.0001, 
                                           'beta_max':0.02, 
                                           'n_steps':1000, 
                                           'mode':'linear' }):
        super().__init__(root_path, category, split, n_points, max_points)
        self.pres = pres
       
        self.noise_scheduler = NoiseSchedulerDDPM(ddpm_params['beta_min'], ddpm_params['beta_max'], ddpm_params['n_steps'], ddpm_params['mode'])

    def __getitem__(self, idx):

        data = super().__getitem__(idx)

        # add noise
        data, t, noise = self.noise_scheduler(data)

        # sparse quantize
        pts = data[:, :3].numpy()
        coords = pts - np.min(pts, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords, self.pres, return_index=True)

        # coordinates as torch tensor
        coords = torch.tensor(coords, dtype=torch.int)
        # features (includes actual coordinates and colors)
        feats = data[indices].float()
        noise = noise[indices].float()
        
        noisy_pts = SparseTensor(coords=coords, feats=feats)
        noise = SparseTensor(coords=coords, feats=noise)
        
        return {'input':noisy_pts, 't':t, 'noise':noise}

def get_sparse_datasets(path, category, pres=1e-5, n_points=2048, max_points=4096,
                        ddpm_params = {'beta_min':0.0001, 
                                       'beta_max':0.02, 
                                       'n_steps':1000, 
                                       'mode':'linear' }):

    tr_dataset = ShapeNetColorSparseNoisy(path, category, split='train', n_points=n_points, max_points=max_points, pres=pres, ddpm_params=ddpm_params) 
    te_dataset = ShapeNetColorSparseNoisy(path, category, split='test' , n_points=n_points, max_points=max_points, pres=pres, ddpm_params=ddpm_params)

    return tr_dataset, te_dataset

def get_sparse_dataloaders(path, category, pres=1e-5, n_points=2048, max_points=4096, batch_size=32, num_workers=8,
                           ddpm_params = {'beta_min':0.0001, 
                                          'beta_max':0.02, 
                                          'n_steps':1000, 
                                          'mode':'linear'}):
    tr_dataset, te_dataset = get_sparse_datasets(path, category=category, pres=pres, n_points=n_points, max_points=max_points, ddpm_params=ddpm_params)

    tr_dl = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=sparse_collate_fn)
    te_dl = DataLoader(te_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=sparse_collate_fn)

    return tr_dl, te_dl

