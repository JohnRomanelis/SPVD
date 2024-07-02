import models.ddpm_unet as ddpm_unet
import models.ddpm_unet_attn as ddpm_unet_attn
import models.spvd as spvd




# SPVD : 32.9M parameters
SPVD = spvd.SPVUnet(point_channels=3, voxel_size=0.1, num_layers=1, pres=1e-5,
                    down_blocks = [[(32, 64, 128, 192, 192, 256), 
                                    (True, True, True, True, False), 
                                    (None, None, None, 8, 8)]], 
                                    # BLOCK 1
                    up_blocks   = [[(256, 192, 192), 
                                    (True, True), 
                                    (8, 8), 
                                    (3, 3)], 
                                    # BLOCK 2
                                   [(192, 128, 64, 32), 
                                    (True, True, False), 
                                    (None, None, None), 
                                    (3, 3, 3)]])

# SPVD-L : 88.1M parameters                 
SPVD_L = spvd.SPVUnet(point_channels=3, voxel_size=0.1, num_layers=1, pres=1e-5,
                    down_blocks = [[(64, 128, 192, 256, 384, 384), 
                                    (True, True, True, True, False), 
                                    (None, None, None, 8, 8)]], 
                                    # BLOCK 1
                    up_blocks   = [[(384, 384, 256), 
                                    (True, True), 
                                    (8, 8), 
                                    (3, 3)], 
                                    # BLOCK 2
                                   [(256, 192, 128, 64), 
                                    (True, True, False), 
                                    (None, None, None), 
                                    (3, 3, 3)]])