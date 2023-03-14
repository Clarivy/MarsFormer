from psbody.mesh import Mesh
from data_loader import load_base_model
import numpy as np
import torch
from util.visualizer import FrameVisulizer

frame_visulizer = FrameVisulizer(
    template_path= "data/USC_neutral.obj"
)

frame_visulizer.visualize(
    frames="data/results/test_2_latest/Ada_OURuv_aligned/chn10.npy",
    output_dir="data/results/test_2_latest/Ada_OURuv_aligned/obj"
)

