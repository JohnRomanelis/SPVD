import gradio as gr
import trimesh
import numpy as np
import torch
import tempfile
import os

from functools import partial
from models.ddpm_unet_attn import SPVUnet
from utils.completion_schedulers import DDPMSparseCompletionSchedulerGPU

COMPLETION_POINTS = 2048
SUPER_RESOLUTION_POINTS = 2048

input_pc = None

get_model = partial(SPVUnet, in_channels=4, voxel_size=0.1, nfs=(32, 64, 128, 256), num_layers=1, attn_chans=8, attn_start=3)

# --- Completion Models --- #

chair_completion_model = get_model()
chair_completion_model.load_state_dict(
    torch.load('checkpoints/CompletionSPVD_S_Chair.pt')['state_dict']
)
chair_completion_model = chair_completion_model.eval().cuda()

table_completion_model = get_model()
table_completion_model.load_state_dict(
    torch.load("checkpoints/CompletionSPVD_S_Table.pt")['state_dict']
)
table_completion_model = table_completion_model.eval().cuda()

# --- Super Resolution Models --- #

chair_super_resolution_model = get_model()
chair_super_resolution_model.load_state_dict(
    torch.load('checkpoints/SuperResolutionSPVD_S_Chair.pt')['state_dict']
)
chair_super_resolution_model = chair_super_resolution_model.eval().cuda()

table_super_resolution_model = get_model()
table_super_resolution_model.load_state_dict(
    torch.load("checkpoints/SuperResolutionSPVD_S_Table.pt")['state_dict']
)
table_super_resolution_model = table_super_resolution_model.eval().cuda()

# --- Scheduler --- #
scheduler = DDPMSparseCompletionSchedulerGPU()

def save_point_cloud_trimesh(filename, points):
    # Create a Trimesh object from the point cloud
    point_cloud_trimesh = trimesh.PointCloud(points)

    # Save the point cloud as a .glb file
    temp_dir = tempfile.mkdtemp()
    output_file = os.path.join(temp_dir, filename)
    point_cloud_trimesh.export(output_file)

    return output_file

def load_input(file):
    # Load the point cloud from the uploaded file
    if file.name.endswith('.ply'):
        mesh = trimesh.load(file.name, file_type='ply')
        points = np.asarray(mesh.vertices)
    elif file.name.endswith('.npy'):
        point_cloud = np.load(file.name)
        points = point_cloud[0] if len(point_cloud.shape) > 2 else point_cloud # remove batch dim
    elif file.name.endswith('.pt'):
        point_cloud = torch.load(file.name).numpy()
        points = point_cloud[0] if len(point_cloud.shape) > 2 else point_cloud # remove batch dim
    else:
        return "Unsupported file format. Please upload a .ply, .npy, or .pt file."

    # Keep the input_pc so that we can use it later and not have to load it again
    global input_pc
    input_pc = points

    output_file = save_point_cloud_trimesh("point_cloud.glb", points)

    return output_file

@torch.no_grad()
def process_point_cloud(task, category):

    pc = torch.tensor(input_pc).float()
  
    if task == 'Completion':
        if category == 'Chair':
            model = chair_completion_model
        else:
            model = table_completion_model

        # Normalize the point cloud - In part completion we use per shape normalization
        pc = (pc - pc.mean(dim=0, keepdim=True))
        pc = pc / pc.std()

        # Add batch dimension
        pc = pc.unsqueeze(0)

        # Complete the point cloud
        res = scheduler.complete(pc, model, n_points=COMPLETION_POINTS, save_process=False)
    
    elif task == 'Super Resolution':
        if category == 'Chair':
            model = chair_super_resolution_model
        else:
            model = table_super_resolution_model

        # Add batch dimension
        pc = pc.unsqueeze(0)

        res = scheduler.complete(pc, model, n_points=SUPER_RESOLUTION_POINTS, save_process=False)

    else: 
        raise NotImplementedError

    points = res.cpu().numpy()
    points = points[0] # remove batch dimension

    output_file = save_point_cloud_trimesh("processed_point_cloud.glb", points)

    return output_file

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown('''
# 🌟 SPVD: Point Cloud Completion & Super Resolution 🚀

Welcome to the SPVD app! This tool is designed to help you explore point cloud **completion** and **super-resolution** tasks effortlessly. 🛠️✨

### 📝 Instructions:
1. 📂 Upload a point cloud file in **.npy** or **.pt** format. You can also select a batch of point clouds, but only the **first point cloud** will be processed.
2. ▶️  Press the **Load Model** button to visualize the input point cloud.
3. 🛠️ Select the task you want to perform: **Completion** or **Super Resolution**.
4. 🪑 Choose the category of the point cloud: **Chair** or **Table**.
5. 🚀 Press the **Process Model** button to visualize the output point cloud.
''')
        gr.Markdown('''        
### 📌 Key Notes:
- **Data Normalization**: 
   - For **part completion**, data is **normalized per shape**. 
   - For **super-resolution**, data is sourced from the **ShapeNet dataset**, normalized **across all shapes** in the dataset. 
   - 📝 **Recommendation**: To reproduce results, we strongly encourage using data from our provided datasets (**PartNet** & **ShapeNet**), especially for the super-resolution task.

- **Model Checkpoints**: 
   - All checkpoints in this demo were produced by training our smallest SPVD model for each task. 
While these checkpoints may not replicate the exact results presented in our main paper, you are encouraged to use them in your research, preferably under the name SPVD-S!

- **Data Visualization**:
   - While Gradio's point cloud visualization tool may not meet publication-quality standards, you can download the processed point cloud and use your preferred framework for high-quality visualization. 🖼️
''')
    with gr.Row():
        gr.Markdown('''                                    
Happy experimenting! 🚀💻✨
  ''')
    with gr.Row():
        # Left Column: File loader, 3D viewer, and parameters
        with gr.Column(scale=1):
            with gr.Row():
                file_loader = gr.File(label="Upload Point Cloud File")
                
            
            with gr.Row():
                task = gr.Radio(['Completion', 'Super Resolution'], label="Task:")

            with gr.Row():
                category = gr.Radio(['Chair', 'Table'], label="Category:") 
        
            with gr.Row():
                load_model_button = gr.Button("Load Model")
                process_model_button = gr.Button("Process Model")

        # Right Column: Results viewer
        with gr.Column(scale=1):
            model_viewer = gr.Model3D(clear_color=(1.0, 1.0, 1.0, 1.0), display_mode="point_cloud", label="Input Point Cloud")
            results_viewer = gr.Model3D(clear_color=(1.0, 1.0, 1.0, 1.0), display_mode="point_cloud", label="Output Point Cloud")

        load_model_button.click(
            load_input,
            inputs=file_loader,
            outputs=[model_viewer]
        )

        process_model_button.click(
            process_point_cloud, 
            inputs=[task, category],
            outputs=[results_viewer]
        )

# Launch the interface
if __name__ == "__main__":
    demo.launch()