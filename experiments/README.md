# Agenda:

- ## Generation 
Contains the design of the DDPM (Denoising Diffusion Probabilistic Models) and DDIM (Denoising Diffusion Implicit Models) scheduler for point cloud generation. 

- ## ModelArchitecture
Includes the design of the SPVD Unet network architecture. There are two model variants available:

- SPVDUnet: A standard variant of the Unet architecture.
- SPVDUnetSymmetric: A symmetric variant of the SPVDUnet architecture.

It also includes the necessary code for training, inference, and testing of the models.

- ## SuperResolution
This notebook contains all the code to train and inference the Super Resolution point cloud model.


- ## PartNetDataset
Contains all the code to generate the **PartNet** dataset used in the paper from the original PartNet data. 


