import torch
import numpy as np
from utils.visualization import quick_vis_batch
from functools import partial
vis_batch = partial(quick_vis_batch, x_offset = 1.5, y_offset=1.5)

pcs = np.load('./results/generated_pcs.npy')[:640]
pcs = pcs.reshape(-1, 32, 2048, 3)

for b in pcs:
    vis_batch(torch.tensor(b))