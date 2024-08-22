# SPVD [arXiv](https://arxiv.org/abs/2408.06145)
Sparse-Point Voxel Diffusion

Installation instructions and documention coming soon! 


# Installation

- ## Installing TorchSparse
1. TorchSparse depends on the [Google Sparse Hash](https://github.com/sparsehash/sparsehash.git) librabry.\
   To install on ubuntu run:
   `sudo apt-get install libsparsehash-dev`

2. Clone the torchsparse repo:\
   `https://github.com/mit-han-lab/torchsparse.git`

3. CD inside the torchsparse directory and run:\
    `pip install -e .`


- ## Install Chamfer Distance and Earth Mover Distance

- ### Chamfer 
1. cd to metrics/chamfer_dist 
2. Run `python setup.py install --user`

- ### EMD
1. cd to metrics/PyTorchEMD
2. Run `python setup.py install`
3. Run `cp ./build/lib.linux-x86_64-cpython-310/emd_cuda.cpython-310-x86_64-linux-gnu.so .`


# Experiments
## Train Generation

## Train Part Completion

## Train Super Resolution
