from psbody.mesh import Mesh
from data_loader import load_base_model
import numpy as np
import torch
# from zlw import align_vertices

# ceo_neutral = Mesh(filename='/data/new_disk/pangbai/FaceFormer/FaceFormer/output_neutral.obj').v[:3931]

raw_template = Mesh(filename = "/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/USC_neutral.obj")
template = raw_template.v / 100
# models = torch.tensor(load_base_model("./data/FLAME", scale=1/100)) - template
# frames = torch.tensor(np.load("/data/new_disk/pangbai/FaceFormer/FaceFormer/demo/result/trim2.npy"))
frames = np.load("/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/results/Ada_OURuv_aligned/chn10.npy").squeeze()
# frames = np.load("/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/GNPFA/valid/chn10/Ada_OURuv_aligned/mesh_pred_all_vs_30fps.npy").squeeze()

# _, R, T, coef = align_vertices(template, ceo_neutral, scale=True)
# models *= coef
# template *= coef

for index, frame in enumerate(frames):
    print(f"Processing {index}")
    frame_mesh = Mesh(v = frame.reshape(-1, 3), f = raw_template.f)
    frame_mesh.write_obj(f"./data/results/Ada_OURuv_aligned/std/{index:06}.obj")
    # result = (frame.reshape(55, 1, 1) * models).sum(0) + template
    # frame_mesh = Mesh(v = result, f = raw_template.f)
    # frame_mesh.write_obj(f"./vis/remap/{index:06}.obj")