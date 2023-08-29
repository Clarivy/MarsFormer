import torch
import pickle
from glob import glob
import os
from psbody.mesh import Mesh
import numpy as np


class PCAInverseTransform(torch.nn.Module):
    def __init__(self, components, mean):
        super(PCAInverseTransform, self).__init__()
        # components = torch.FloatTensor(components)
        # mean = torch.FloatTensor(mean)
        self.components = torch.nn.Parameter(components, requires_grad=False)
        self.mean = torch.nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        return x @ self.components + self.mean


def get_inverse_model(pca_path):
    with open(pca_path, "rb") as f:
        ipca = pickle.load(f)
    components_torch = torch.tensor(ipca.components_, dtype=torch.float)
    mean_torch = torch.tensor(ipca.mean_, dtype=torch.float)
    torch_pca = PCAInverseTransform(components=components_torch, mean=mean_torch)

    return torch_pca


def load_vertices(path, scale=1):
    return Mesh(filename=path).v * scale


def load_base_model(path, scale=1):
    meshes = []
    for filename in sorted(glob(os.path.join(path, "*.obj"))):
        meshes.append(load_vertices(filename, scale=scale))
    return np.array(meshes)
