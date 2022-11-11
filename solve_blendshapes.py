from psbody.mesh import Mesh
import numpy as np
from zlw import iterative_solve_blendshapes, align_vertices
from data_loader import load_base_model
from tqdm import tqdm
import zlw

ceo_neutral = Mesh(filename='/data/new_disk/pangbai/FaceFormer/FaceFormer/output_neutral.obj').v[:3931]

print("Reading template...")
raw_template = Mesh(filename="/data/new_disk/pangbai/FaceFormer/FaceFormer/data/000_generic_neutral_mesh.obj")
template = raw_template.v / 100

print("Reading base models...")
base_models = load_base_model("./data/FLAME", scale=1/100) - template
print(base_models.shape)

print("Loading frame data...")
source_frames = np.load("./demo/result/unityceo.npy")
source_frames = source_frames.reshape(-1, 5023, 3)
source_frames = source_frames[:, :3931, :]
# indice = [i for i in range(source_frames.shape[0])]
output_frames = []

_, R, T, coef = align_vertices(template, ceo_neutral, scale=True)

base_models *= coef
template *= coef

print("Executing transformation...")

count = 0
for frame in tqdm(source_frames):
    # Mesh(v=frame, f=raw_template.f).write_obj(f"./vis/origin/{count:06d}.obj")
    output_frame = iterative_solve_blendshapes(frame, template, base_models)[0]

    output_frames.append(output_frame)
    count += 1
    if count % 50 == 0:
        np.save("./BS_diffuse_checkpoint.npy", np.array(output_frames))

    # for result in output_frame:
    #     print(result.shape)
    # exit()

np.save("./BS_diffuse.npy", np.array(output_frames))

# template = Mesh(filename='/data/new_disk/pangbai/FaceFormer/FaceFormer/data/USC/000_generic_neutral_mesh.obj')
# biwi = Mesh(filename='/data/new_disk/pangbai/FaceFormer/FaceFormer/BIWI/templates/BIWI.ply') 
# voca = Mesh(filename='/data/new_disk/pangbai/FaceFormer/FaceFormer/vocaset/FaceTalk_170725_00137_TA/sentence01/sentence01.000001.ply') 
# flame = Mesh(filename='/data/new_disk/pangbai/FaceFormer/FaceFormer/data/FLAME/000_generic_neutral_mesh.obj') 

# Get the mesh of the first 3931 vertices

# flame_clipped = Mesh(v=flame.v[:3931], f=flame.f)
# flame_clipped.write_obj("./noeyes.obj")

# print("ZLW", template.v.shape)
# print("BIWI", biwi.v.shape)
# print("VOCA", voca.v.shape)
# print("FLAME", flame.v.shape)