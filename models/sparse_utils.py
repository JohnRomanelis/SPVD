import numpy as np
import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn.functional as spf
from torchsparse import SparseTensor
from torchsparse.nn.functional.devoxelize import calc_ti_weights
from torchsparse.nn.utils import *
from torchsparse.utils import *
from torchsparse.utils.tensor_cache import TensorCache
import torch_scatter
from typing import List, Union, Tuple
from itertools import repeat

__all__ = ["initial_voxelize", "point_to_voxel", "voxel_to_point", "PointTensor", "unique", 
           "ravel_hash_torch", "sparse_quantize_torch", "batched_ravel_hash_torch", "batch_sparse_quantize_torch"]

# The initial section of this code is reused from the SPVNAS project's original codebase,
# available at https://github.com/mit-han-lab/spvnas/blob/master/core/models/utils.py.
# Below, I introduce my own code to transform a BxNx3 tensor directly into a SparseTensor,
# eliminating the use of numpy and thereby avoiding the need to transfer data to the CPU.

class PointTensor(SparseTensor):
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 1,
    ):
        super().__init__(feats=feats, coords=coords, stride=stride)
        self._caches.idx_query = dict()
        self._caches.idx_query_devox = dict()
        self._caches.weights_devox = dict()


def sphashquery(query, target, kernel_size=1):
    hashmap_keys = torch.zeros(
        2 * target.shape[0], dtype=torch.int64, device=target.device
    )
    hashmap_vals = torch.zeros(
        2 * target.shape[0], dtype=torch.int32, device=target.device
    )
    hashmap = torchsparse.backend.GPUHashTable(hashmap_keys, hashmap_vals)
    hashmap.insert_coords(target[:, [1, 2, 3, 0]])
    kernel_size = make_ntuple(kernel_size, 3)
    kernel_volume = np.prod(kernel_size)
    kernel_size = make_tensor(kernel_size, device=target.device, dtype=torch.int32)
    stride = make_tensor((1, 1, 1), device=target.device, dtype=torch.int32)
    results = (
        hashmap.lookup_coords(
            query[:, [1, 2, 3, 0]], kernel_size, stride, kernel_volume
        )
        - 1
    )[: query.shape[0]]
    return results


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [z.C[:, 0].view(-1, 1), (z.C[:, 1:] * init_res) / after_res], 1
    )
    # optimization TBD: init_res = after_res
    new_int_coord = torch.floor(new_float_coord).int()
    sparse_coord = torch.unique(new_int_coord, dim=0)
    idx_query = sphashquery(new_int_coord, sparse_coord).reshape(-1)

    sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
    new_tensor = SparseTensor(sparse_feat, sparse_coord, 1)
    z._caches.idx_query[z.s] = idx_query
    z.C = new_float_coord
    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z._caches.idx_query.get(x.s) is None:
        # Note: x.C has a smaller range after downsampling.
        new_int_coord = torch.cat(
            [
                z.C[:, 0].int().view(-1, 1),
                torch.floor(z.C[:, 1:] / x.s[0]).int(),
            ],
            1,
        )
        idx_query = sphashquery(new_int_coord, x.C)
        z._caches.idx_query[x.s] = idx_query
    else:
        idx_query = z._caches.idx_query[x.s]
    # Haotian: This impl. is not elegant
    idx_query = idx_query.clamp_(0)
    sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
    new_tensor = SparseTensor(sparse_feat, x.C, x.s)
    new_tensor._caches = x._caches

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if (
        z._caches.idx_query_devox.get(x.s) is None
        or z._caches.weights_devox.get(x.s) is None
    ):
        point_coords_float = torch.cat(
            [z.C[:, 0].int().view(-1, 1), z.C[:, 1:] / x.s[0]],
            1,
        )
        point_coords_int = torch.floor(point_coords_float).int()
        idx_query = sphashquery(point_coords_int, x.C, kernel_size=2)
        weights = calc_ti_weights(point_coords_float[:, 1:], idx_query, scale=1)

        if nearest:
            weights[:, 1:] = 0.0
            idx_query[:, 1:] = -1
        new_feat = spf.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches
        new_tensor._caches.idx_query_devox[x.s] = idx_query
        new_tensor._caches.weights_devox[x.s] = weights
        z._caches.idx_query_devox[x.s] = idx_query
        z._caches.weights_devox[x.s] = weights

    else:
        new_feat = spf.spdevoxelize(
            x.F, z._caches.idx_query_devox.get(x.s), z._caches.weights_devox.get(x.s)
        )
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches

    return new_tensor


### New code starts here ###
def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)

@torch.no_grad()
def ravel_hash_torch(x):
    device = x.device
    assert x.ndim == 2, x.shape

    x = x - x.min(dim=0).values
    x = x.long() # This commmand is not equivalent but the int64 range should cover our needs. PyTorch support for uint64 is limited. 
    xmax = x.max(dim=0).values.long() + 1

    h = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def sparse_quantize_torch(
    coords,
    voxel_size: Union[float, Tuple[float, ...]] = 1,
    *,
    return_index: bool = False,
):
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = torch.tensor(voxel_size)
    coords = torch.floor(coords / voxel_size).int() # torch.int = torch.int32

    _, indices = unique(
        ravel_hash_torch(coords), dim=0
    )
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]

    return outputs[0] if len(outputs) == 1 else outputs

@torch.no_grad()
def batched_ravel_hash_torch(x):
    device = x.device
    #assert x.ndim == 2, x.shape
    x = x - x.min(dim=1, keepdims=True).values
    x = x.long() # This commmand is not equivalent but the int64 range should cover our needs. PyTorch support for uint64 is limited. 
    xmax = x.max(dim=1, keepdims=True).values.long() + 1

    h = torch.zeros((x.shape[0],x.shape[1]), dtype=torch.long).to(x.device)
    for k in range(x.shape[2] - 1):
        h[:] += x[:, :, k]
        h[:] *= xmax[:, :, k + 1]
    h[:] += x[:, :, -1]
    return h

def batch_sparse_quantize_torch(
    coords, # B x N x 3
    voxel_size: Union[float, Tuple[float, ...]] = 1,
    *,
    batch_index = None,
    return_index: bool = False,
    return_batch_index = True
):
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    B, N, C = coords.shape
    
    if batch_index == None:
        batch_index = torch.arange(0, B).repeat_interleave(N).unsqueeze(-1).to(coords.device)
        
    voxel_size = torch.tensor(voxel_size, device=coords.device)
    coords = torch.floor(coords / voxel_size).int() # torch.int = torch.int32

    hashed_coords = batched_ravel_hash_torch(coords) # B x N
    hashed_coords = hashed_coords.view(-1, 1)
    
    batched_hashed_coords = torch.cat([batch_index, hashed_coords], dim=-1)
    #print(batched_hashed_coords.shape)
    
    _, indices = unique(
        batched_hashed_coords, dim=0
    )

    coords = coords.view(-1, C)
    coords = torch.cat([batch_index, coords], dim=-1)
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]

    if return_batch_index:
        outputs += [batch_index]
        
    return outputs[0] if len(outputs) == 1 else outputs


class Torch2Torchsparse:

    def __init__(self, pres=1e-5, tensorType=SparseTensor):
        self.pres = pres
        assert isinstance(tensorType, (SparseTensor, PointTensor))

    def __call__(self, pc, f=None):
        # Point Cloud is of shape B x N x F where the first 3 columns are the coordinates
        # f contains additional features that may require quantization 

        # Point Coordinates should be Positive
        coords = pc[..., :3] # In case points have additional features
        coords = coords - coords.min(dim=1, keepdim=True).values
        coords, indices = batch_sparse_quantize_torch(coords, voxel_size=self.pres, return_index=True, return_batch_index=False)
        feats = pc.view(-1, 3)[indices]

        if f is not None:
            if not isinstance(f, (tuple, list)): f = [f]
            f = [ff.view(-1, 3)[indices] for ff in f]
            if len(f) == 1: f = f[0]
        
        return PointTensor(coords=coords, feats=feats).to(coords.device) if f is None \
            else (PointTensor(coords=coords, feats=feats).to(coords.device), f)