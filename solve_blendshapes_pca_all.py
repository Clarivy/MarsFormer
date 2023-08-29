# %%
from psbody.mesh import Mesh
import numpy as np
from zlw import iterative_solve_blendshapes, align_vertices
from data_loader import load_base_model
from tqdm import tqdm
import zlw
import sklearn
import numpy as np
import matplotlib.pyplot as plt


# %%
from glob import glob
import os

# %%
target_files = list(glob("/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/GNPFA_*/train/*/TaylorSwift/mesh_pred_all_vs_30fps.npy"))
target_files = sorted(
    target_files,
    key=lambda x: 0 if "NEU" in x else 1
)

# %%
print("Reading data template...")
data_template = np.load("/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/GNPFA_CREMA-D/identity/TaylorSwift.npy")
data_template = data_template.flatten()
print(data_template.shape)

# %%
print(data_template.max())
print(data_template.min())
print(data_template.mean())

# %%
sample_data = np.load(target_files[0])
print(sample_data.max())
print(sample_data.min())
print(sample_data.mean())

# %%
from sklearn.decomposition import IncrementalPCA
batch_size = 1024
latent_dim = 64

ipca = IncrementalPCA(n_components=latent_dim, batch_size=batch_size)

# %%
last_frames = None
for target_file in tqdm(target_files):
    source_frames = np.load(target_file).reshape(-1, 42186)
    if last_frames is not None:
        source_frames = np.concatenate((last_frames, source_frames), axis=0)
        last_frames = None
    if source_frames.shape[0] < batch_size:
        last_frames = source_frames
        continue
    ipca.partial_fit(source_frames - data_template)

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

# %%
data_list = []
for target_file in tqdm(target_files):
    target_data = np.load(target_file).reshape(-1, 42186)
    data_list.append(ipca.transform(target_data - data_template))

# %%
print((source_frames - data_template).std())
print((source_frames - data_template).mean())
print(ipca.transform(source_frames - data_template).std())
print(ipca.transform(source_frames - data_template).mean())

# %%
scaler.fit(np.concatenate(data_list, axis=0))

import pickle

with open("ipca_diff_dataall_model.pkl", "wb") as f:
    pickle.dump(ipca, f)

with open("ipca_diff_norm_model.pkl", "wb") as f:
    pickle.dump(scaler, f)


# %%
for target_file, data_item in zip(target_files, data_list):
    normalized_data = scaler.transform(data_item)
    output_file = os.path.join(os.path.dirname(target_file), "pca_norm_alldata_30fps.npy")
    np.save(output_file, normalized_data)

# %%
import random

# %%
sample_files = random.sample(target_files, 100)
mse_max_results = []
mse_mean_results = []

for target_file in tqdm(sample_files):
    sample_data = np.load(target_file).reshape(-1, 42186)
    reduced_data = ipca.transform(sample_data - data_template)
    recon_data = ipca.inverse_transform(reduced_data)
    mse_mean_results.append(
        (((recon_data - sample_data + data_template) / 100) ** 2).mean()
    )
    mse_max_results.append(
        (recon_data - sample_data + data_template).max()    
    )

mse_max = max(mse_max_results)
mse_mean = np.mean(mse_mean_results)

# %%
print("Max", mse_max)
print("Mean", mse_mean)

# %%
reduced_data = ipca.transform(sample_data - data_template)
reduced_data.shape
print(reduced_data.max())
print(reduced_data.min())
print((sample_data - data_template).max())
print((sample_data - data_template).min())

# %%
recon_data = ipca.inverse_transform(reduced_data)
recon_data.shape

# %%
print((((recon_data + data_template - sample_data) / 100) ** 2).mean())
print((recon_data + data_template - sample_data).max())
