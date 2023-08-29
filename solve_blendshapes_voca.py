from psbody.mesh import Mesh
import numpy as np
from zlw import iterative_solve_blendshapes, align_vertices
from data_loader import load_base_model
from tqdm import tqdm
import scipy.optimize as opt

from glob import glob
import os

from multiprocessing.pool import ThreadPool
import multiprocessing
from joblib import Parallel, delayed
import pickle

print("Reading template...")
raw_template = Mesh(filename="/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/000_generic_neutral_mesh_flame.obj")
template = raw_template.v
print(template.mean())
print(template.max())

print("Reading base models...")
base_models = load_base_model("./data/FLAME", scale=1)
print(base_models[0].mean())
print(base_models[0].max())
base_models -= template

base_models = base_models.reshape(55, -1, 3)
template = template.reshape(-1, 3)

with open("/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/vocaset/templates.pkl", "rb") as f:
    data_templates:dict = pickle.load(f, encoding='latin1')

data_templates = {
    k: v * 100
    for k, v in data_templates.items()
}

# pool = ThreadPool(20)
# pool = multiprocessing.Pool(processes = 20)
def solve(target_file):
    npy_name = os.path.basename(target_file)
    subject_id = "_".join(npy_name.split("_")[:-1])
    print(f"Processing {target_file}")
    output_frames = []
    recons = []
    source_frames = np.load(target_file)[:,:11793].reshape(-1, 3931, 3) * 100
    print(source_frames.shape)
    for frame in tqdm(source_frames):
        output_frame = iterative_solve_blendshapes(frame, data_templates[subject_id][:3931], base_models, min_max=(0, 1), debug=False)[0]
        output_frames.append(output_frame)
        recons.append(data_templates[subject_id][:3931] + np.sum(output_frame.reshape(55, 1, 1)*base_models, axis=0))
    # count loss
    print(f"mean : {(((recons-source_frames)/100)**2).mean()}")
    print(f"max : {(((recons-source_frames)/100)).max()}")

    output_file = os.path.join("vocaset", npy_name)
    np.save(output_file, np.array(output_frames))
    print(output_file)

def dist(x, frame):
    recon = (x@base_models.reshape(55, -1) + template.reshape(-1)).reshape(-1, 3)
    return (np.linalg.norm(recon - frame)**2 / len(recon))**0.5

if __name__ == '__main__':
    target_files = sorted(list(glob("/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/vocaset/vertices_npy/*.npy")))
    # pool.map(solve, target_files)
    results = Parallel(n_jobs=-1)(delayed(solve)(target_file) for target_file in target_files)
    # solve(target_files[0])