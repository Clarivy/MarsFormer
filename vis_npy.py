from psbody.mesh import Mesh
from data_loader import load_base_model
import numpy as np
import torch
from util.visualizer import FrameVisulizer

frames = np.load("demo/result/zqx_clip.npy")
# template = np.load("data/GNPFA_CREMA-D/identity/000_generic_neutral_mesh_usc.npy")

# frames += template.flatten()

frame_visulizer = FrameVisulizer(
    template_path= "data/000_generic_neutral_mesh_usc.obj"
)

frame_visulizer.visualize(
    frames=frames,
    output_dir="data/results/zqx_clip"
)


# frame_visulizer = FrameVisulizer(
#     template_path= "data/000_generic_neutral_mesh_usc.obj"
# )

# frame_visulizer.visualize(
#     frames="/data/new_disk/new_disk/pangbai/FaceFormer/motion-diffusion-model/results/npy/cremad_nue+voca+chn_000510000_maggie5_vert.npy",
#     output_dir="data/results/cremad_nue+voca+chn_000510000_maggie5_vert"
# )


# frame_visulizer = FrameVisulizer(
#     template_path= "data/000_generic_neutral_mesh_flame.obj"
# )

# frame_visulizer.visualize(
#     frames="test.npy",
#     output_dir="data/results/bs_voca"
# )