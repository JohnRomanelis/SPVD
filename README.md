# SPVD [arXiv](https://arxiv.org/abs/2408.06145)
Sparse-Point Voxel Diffusion

Installation instructions and documention coming soon! 

# News:

- **12/8/2024**: Arxiv submission of the SPVD preprint.

# Installation

### 1. Set Up an Anaconda Environment

We recommend using Anaconda to manage your Python environment.

```
conda create --name spvd python=3.9
conda activate spvd
```

### 2. Clone the Repository

```
git clone https://github.com/JohnRomanelis/SPVD.git
```

### 3. Install PyTorch

We have tested our code with PyTorch 2.0 and CUDA 11.8. You can install the compatible version using the following command:

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. Install pclab

*pclab* is an helper library, based on the fast.ai [Practical Deep Learning-Part 2](https://course.fast.ai/Lessons/part2.html) course. 

**Note**: Make sure you have installed PyTorch before install pclab to make sure you install the correct version.

1. Clone the *pclab* repository.
```
git clone https://github.com/JohnRomanelis/pclab.git
```
2. Navigate into the *pclab* directory:
```
cd pclab
```
3. Install *pclab*. This will automatically install the required dependencies:
```
pip install -e .
```



### 5. Installing TorchSparse
1. TorchSparse depends on the [Google Sparse Hash](https://github.com/sparsehash/sparsehash.git) librabry.\
   To install on ubuntu run:
   `sudo apt-get install libsparsehash-dev`

2. Clone the torchsparse repo:\
   `https://github.com/mit-han-lab/torchsparse.git`

3. CD inside the torchsparse directory and run:\
    `pip install -e .`


### 6. Install Chamfer Distance and Earth Mover Distance

- **Chamfer** 
1. cd to metrics/chamfer_dist 
2. Run `python setup.py install --user`

- **EMD**
1. cd to metrics/PyTorchEMD
2. Run `python setup.py install`
3. Run `cp ./build/lib.linux-x86_64-cpython-310/emd_cuda.cpython-310-x86_64-linux-gnu.so .`


# Experiments
## Train Generation

## Train Part Completion

## Train Super Resolution
