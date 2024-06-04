import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse
import torchsparse.nn as spnn

import fastcore.all as fc

from .modules import *
from .sparse_utils import *
from .utils import *

class VoxelResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ks=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=ks, stride=1),
            spnn.BatchNorm(out_channels),
            spnn.ReLU()
        )

        self.conv2 = nn.Sequential(
            spnn.Conv3d(out_channels, out_channels, kernel_size=ks, stride=1),
            spnn.BatchNorm(out_channels)
        )

        self.out_relu = spnn.ReLU()

        self.identity = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else fc.noop

    def forward(self, x_in):
        # convolutions
        x = self.conv2(self.conv1(x_in))
        
        # skip connection
        x.F = x.F + self.identity(x_in.F)

        return self.out_relu(x)


class SimpleDownBlock(nn.Module):

    def __init__(self, n_emb, ni, nf):
        super().__init__()

        self.t_emb = TimeEmbeddingBlock(n_emb, ni)

        self.conv = nn.Sequential(
            spnn.Conv3d(ni, ni, kernel_size=3),
            spnn.BatchNorm(ni),
            spnn.ReLU()
        )

        self.down_conv = nn.Sequential(
            spnn.Conv3d(ni, nf, kernel_size=2, stride=2),
            spnn.BatchNorm(nf),
            spnn.ReLU()
        )

    def forward(self, x, t):
        # include time embedding information
        x.F = self.t_emb(x.F, t, x.C[:, 0])
        
        # convolutions
        x = self.conv(x)
        x = self.down_conv(x)
        return x
    
class SimpleUpBlock(nn.Module):

    def __init__(self, n_emb, ni, nf):
        super().__init__()

        self.t_emb = TimeEmbeddingBlock(n_emb, ni//2)

        self.feature_fusion = nn.Sequential(
            spnn.Conv3d(ni, ni//2, kernel_size=1),
            spnn.BatchNorm(ni//2),
            spnn.ReLU()
        )

        self.conv = nn.Sequential(
            spnn.Conv3d(ni//2, ni//2, kernel_size=3),
            spnn.BatchNorm(ni//2),
            spnn.ReLU()
        )

        self.up_conv = nn.Sequential(
            spnn.Conv3d(ni//2, nf, kernel_size=2, stride=2, transposed=True),
            spnn.BatchNorm(nf),
            spnn.ReLU()
        )
    
    def forward(self, x, t):
        # feature fusion
        x = self.feature_fusion(x)

        # include time embedding information
        x.F = self.t_emb(x.F, t, x.C[:, 0])
        
        # convolutions
        x = self.conv(x)
        x = self.up_conv(x)
        return x

class SPVUnet(nn.Module):

    def __init__(self, voxel_size, pres=1e-5):
        super().__init__()
        self.pres, self.voxel_size = pres, voxel_size

        # time embedding related code 
        self.n_temb = nf = 32
        n_emb = nf*4
        self.emb_mlp = nn.Sequential(lin(self.n_temb, n_emb, norm=nn.BatchNorm1d),
                                     lin(n_emb, n_emb))
        

        # Stem Conv
        self.stem_conv = VoxelResBlock(3, 32)

        # Down Conv 1
        self.down_conv1 = SimpleDownBlock(n_emb, 32, 64)

        # Down Conv 2
        self.down_conv2 = SimpleDownBlock(n_emb, 64, 128)

        # Down Conv 3
        self.down_conv3 = SimpleDownBlock(n_emb, 128, 256)

        # Down Conv 4
        self.down_conv4 = SimpleDownBlock(n_emb, 256, 512)


        self.up_conv0 = SimpleUpBlock(n_emb, 2 * 512, 256)

        # Up Conv 1
        self.up_conv1 = SimpleUpBlock(n_emb, 2 * 256, 128)

        # Up Conv 2 
        self.up_conv2 = SimpleUpBlock(n_emb, 2 * 128, 64)

        # Up Conv 3
        self.up_conv3 = SimpleUpBlock(n_emb, 2 * 64, 32)

        # Point MLPs
        self.PointMLP1 = nn.ModuleList([
            nn.Linear(32, 128),
            nn.Sequential(nn.BatchNorm1d(128), nn.ReLU())
        ])

        self.PointMLP2 = nn.ModuleList([
            nn.Linear(128, 512),
            nn.Sequential(nn.BatchNorm1d(512), nn.ReLU())
        ])

        self.PointMLP3 = nn.ModuleList([
            nn.Linear(512, 128),
            nn.Sequential(nn.BatchNorm1d(128), nn.ReLU())
        ])

        self.PointMLP4 = nn.ModuleList([
            nn.Linear(128, 32),
            nn.Sequential(nn.BatchNorm1d(32), nn.ReLU())
        ])

        # Output UNET covn
        self.out_unet = nn.Sequential(
            spnn.Conv3d(2 * 32, 32, kernel_size=3),
            spnn.BatchNorm(32),
            spnn.ReLU()
        )

        # Output Conv
        self.out_conv = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )


    def forward(self, inp):

        x, t = inp

        # 0. Time Embedding
        z = PointTensor(x.F, x.C.float())
        t = timestep_embedding(t, self.n_temb)
        emb = self.emb_mlp(t)

        # 1. Initial Voxelization
        x0 = initial_voxelize(z, self.pres, self.voxel_size)

        # 2. Stem Conv
        x1 = self.stem_conv(x0)
        
        # 3. First Point Stage
        # After the stem convolution, we just want to save the point features
        z1 = voxel_to_point(x1, z)

        # 4. Down Conv 1 | 34 -> 64
        x2 = self.down_conv1(x1, emb)

        # 5. Down Conv 2 | 64 -> 128
        x3 = self.down_conv2(x2, emb)

        # 6. Second Point Stage
        z2 = voxel_to_point(x3, z1)
        z2.F = self.PointMLP1[1](z2.F + self.PointMLP1[0](z1.F))
        x3p = point_to_voxel(x3, z2)

        # 5. Down Conv 3
        x4 = self.down_conv3(x3p, emb)

        # 5.5 Down Conv 4
        x4n = self.down_conv4(x4, emb)

        # 6. Third Point Stage
        z3 = voxel_to_point(x4n, z2)
        z3.F = self.PointMLP2[1](z3.F + self.PointMLP2[0](z2.F))
        x4p = point_to_voxel(x4n, z3)


        # 7. Up Conv 0
        x4n = self.up_conv0(torchsparse.cat([x4p, x4n]), emb)

        # 7. Up Conv 1
        x5 = self.up_conv1(torchsparse.cat([x4, x4n]), emb)
        
        # 8. Fourth Point Stage
        z4 = voxel_to_point(x5, z3)
        z4.F = self.PointMLP3[1](z4.F + self.PointMLP3[0](z3.F))
        x5p = point_to_voxel(x5, z4)

        # 9. Up Conv 2
        x6 = self.up_conv2(torchsparse.cat([x5p, x3p]), emb)

        # 10. Up Conv 3
        x7 = self.up_conv3(torchsparse.cat([x6, x2]), emb)

        # 11. Fifth Point Stage
        z5 = voxel_to_point(x7, z4)
        z5.F = self.PointMLP4[1](z5.F + self.PointMLP4[0](z4.F))
        x5p = point_to_voxel(x7, z5)

        # 12. Output UNET Conv
        x8 = self.out_unet(torchsparse.cat([x5p, x1]))

        # 13. Last Point Stage
        z6 = voxel_to_point(x8, z5)

        # 9. Output Conv
        out = self.out_conv(z6.F)

        return out

        




