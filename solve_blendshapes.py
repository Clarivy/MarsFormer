from psbody.mesh import Mesh
import numpy as np
from zlw import iterative_solve_blendshapes, align_vertices
# from data_loader import load_base_model
from tqdm import tqdm
import scipy.optimize as opt

from glob import glob
import os

from multiprocessing.pool import ThreadPool
import multiprocessing
from joblib import Parallel, delayed

# multiprocessing.set_start_method('forkserver')
def load_vertices(path, scale = 1):
    return Mesh(filename=path).v * scale

def load_base_model(path, scale = 1):
    import glob

    if path == "":
        return None

    meshes = []
    for filename in sorted(glob.glob(os.path.join(path, "*.obj"))):
        meshes.append(load_vertices(filename, scale=scale))
    return meshes

print("Reading template...")
raw_template = Mesh(filename="./data/000_generic_neutral_mesh_usc.obj")
template = raw_template.v
print(template.shape)
print(template.mean())

print("Reading base models...")
base_models = load_base_model("./data/USC", scale=1) - template
print(base_models.shape)
print(base_models.max())

print("Reading data template...")
# data_template = np.load("./data/GNPFA_CREMA-D/identity/000_generic_neutral_mesh_usc.npy")
data_template = np.load("/home/zhaoqch1/Audio2Face3D/data/lpy/GNPFA/GNPFA2pack/output/identity/000_generic_neutral_mesh_usc.npy")

data_template = data_template.squeeze()
print(data_template.shape)

_, R, T, coef = align_vertices(template, data_template, scale=True)
base_models *= coef
template *= coef
print(coef)

# pool = ThreadPool(20)


def solve(target_file):
    print(f"Processing {target_file}")
    output_frames = []
    recons = []
    source_frames = np.load(target_file)
    for frame in tqdm(source_frames):
        output_frame = iterative_solve_blendshapes(frame, template, base_models, min_max=(0, 1), debug=False)[0]
        # output_frame = opt.minimize(lambda x: dist(x, frame), np.zeros(55), method="L-BFGS-B", bounds=[(0, 1)]*55).x

        output_frames.append(output_frame)
        recons.append(template + np.sum(output_frame.reshape(55, 1, 1)*base_models, axis=0))

    # count loss
    print(f"mean : {(((recons-source_frames)/100)**2).mean()}")
    print(f"max : {(((recons-source_frames)/100)).max()}")

    output_file = os.path.join(os.path.dirname(target_file), "bs_30fps.npy")
    np.save(output_file, np.array(output_frames))
    print(output_file)

def dist(x, frame):
    recon = (x@base_models.reshape(55, -1) + template.reshape(-1)).reshape(-1, 3)
    return (np.linalg.norm(recon - frame)**2 / len(recon))**0.5



if __name__ == '__main__':
    # pool = multiprocessing.Pool(processes = 40)
    # target_files = sorted(glob("./data/GNPFA_MEAD/train/*disgusted*/000_generic_neutral_mesh_usc/mesh_pred_all_vs_30fps.npy"))
    target_files = sorted(glob("/home/zhaoqch1/Audio2Face3D/data/lpy/GNPFA/data_output_RAVDESS/*/000_generic_neutral_mesh_usc/mesh_pred_all_vs_30fps.npy"))
    #ongoing ravdess

    # print(target_files)
    print(len(target_files))
    remove=[]
    for target_file in target_files:
        if os.path.exists(os.path.join(os.path.dirname(target_file), "bs_30fps.npy")):
            remove.append(target_file)
    target_files = sorted(list(set(target_files) - set(remove)))
    print(len(target_files))

    # pool.map_async(solve, target_files)
    # pool.close()
    # pool.join()
    results = Parallel(n_jobs=4)(delayed(solve)(target_file) for target_file in target_files)
    # solve(target_files[0])