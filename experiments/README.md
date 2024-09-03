# Agenda:

- ## GenerationSchedulers 
Contains the design of the DDPM (Denoising Diffusion Probabilistic Models) and DDIM (Denoising Diffusion Implicit Models) scheduler for point cloud generation. 

- ## ModelArchitecture
Includes the design of the SPVD Unet network architecture. There are two model variants available:

- SPVDUnet: A standard variant of the Unet architecture.
- SPVDUnetSymmetric: A symmetric variant of the SPVDUnet architecture.

It also includes the necessary code for training, inference, and testing of the models.

- ## TrainGeneration
Train a generative model for unconditional point cloud generation in a single class of ShapeNet.

- ## ConditionalGeneration
Train a conditional model on all categories of ShapeNet.

- ## PartNetDataset
Contains all the code to generate the **PartNet** dataset used in the paper from the original PartNet data. 

- ## TrainCompletion
This notebook contains all the code needed to train and perform inference with the Completion point cloud model.

- ## SuperResolution
This notebook contains all the code needed to train and perform inference with the Super Resolution point cloud model.

- ## GenerationLowRes
Exploring a network architecture for generating point clouds with lower number of points, enabling attention layers at earlier stages. 

- ## LowResGeneration&SuperRes
Generation of low resolution point clouds, with the aim of achieving higher shape diversity, followed by a super resolution network to achieve the required detail.

