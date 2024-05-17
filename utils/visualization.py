import open3d as o3d
from pclab.utils import pc_to_o3d
import numpy as np

from matplotlib import colormaps
cmap = colormaps['jet']

def quick_vis_batch(batch, grid=(8, 4), x_offset=2.5, y_offset=2.5):
    
    batch = batch.detach().cpu().clone()
    
    assert len(grid) == 2
    
    if batch.shape[0] <= np.prod(grid): batch = batch[:np.prod(grid)]
            
    x_offset_start = - x_offset * grid[0] // 2
    x_offset_start = x_offset_start + x_offset / 2 if grid[0] % 2 == 0 else x_offset_start
    
    y_offset_start = - y_offset * grid[1] // 2
    y_offset_start = y_offset_start + y_offset / 2 if grid[1] % 2 == 0 else y_offset_start
    
    pcts = []
    
    k=0
    for i in range(grid[0]):
        for j in range(grid[1]):
            
            # get point cloud to cpu
            pc = batch[k]
            
            # translate the point cloud properly
            pc[:, 0] += x_offset_start + i * x_offset
            pc[:, 1] += y_offset_start + j * y_offset
            
            # turn in into an open3d point cloud
            pct = pc_to_o3d(pc)
            
            # append it to the pcts list
            pcts.append(pct)
            
            # incriment k
            k+=1
            
        if k > batch.shape[0]-1: break
            
    o3d.visualization.draw_geometries(pcts)

def vis_pc_sphere(pc, radius=0.1, resolution=30, color=None):
    
    pc = pc.detach().cpu().squeeze().numpy()

    if color is None:
        # sample a colormap based on the z-direction of the point cloud
        color_val = pc[:, -1]
        # normalize the color values in range [0, 1]
        color_val = (color_val - color_val.min()) / (color_val.max() - color_val.min())
        # get the color 
        color = cmap(color_val)[:, :3]

    
    # create a mesh that will contain all the spheres
    mesh = o3d.geometry.TriangleMesh()
    # create a sphere for each point in the point cloud
    for i, p in enumerate(pc):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution).translate(p)
        sphere.paint_uniform_color(color[i])
        mesh += sphere

    o3d.visualization.draw_geometries([mesh])