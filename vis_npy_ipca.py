from psbody.mesh import Mesh
from data_loader import load_base_model
import numpy as np
from util.visualizer import FrameVisulizer
import pickle

test_data = np.load("/data/new_disk/new_disk/pangbai/FaceFormer/motion-diffusion-model/results/pca_norm_ml160_vert_vel_000600000_maggie5.npy")

with open("data/ipca_diff_model.pkl", "rb") as f:
    ipca = pickle.load(f)
with open("data/ipca_diff_norm_model.pkl", "rb") as f:
    scaler = pickle.load(f)

data_template = np.load("/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/GNPFA_CREMA-D/identity/TaylorSwift.npy")
data_template = data_template.flatten()

def get_vert(x):
    return ipca.inverse_transform(scaler.inverse_transform(x))

def get_vel(vert):
    return vert[1:] - vert[:-1]

recon_test = get_vert(test_data) + data_template

frame_visulizer = FrameVisulizer(
    template_path= "data/GNPFA_CREMA-D/identity/TaylorSwift.obj"
)

frame_visulizer.visualize(
    frames=recon_test,
    output_dir="data/results/diff_maggie5"
)

