import os
import numpy as np
from glob import glob
import numpy as np
import os
import trimesh
import pickle

from multiprocessing import Pool

from zlw.contrib.zyh_topo_transfer.topology_transfer import ObjHandle, trimesh, applyTransfer, readPointOffset

template_keys = ['FaceTalk_170904_00128_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA', 'FaceTalk_170904_03276_TA']

source_handle = ObjHandle("vocaset/templates/FLAME_sample.obj", triangulate=True)


def load_maps():
    map_dict = {}
    for template_key in template_keys:
        map_path = os.path.join("vocaset/trans_map", template_key + ".map")
        map_dict[template_key] = readPointOffset(map_path)
    return map_dict

def trans_vertices(vertices: np.ndarray, trans_map):
    closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv, faces, tex_coords, face_tcs = trans_map
    vertices = vertices.reshape(5023, 3)
    source = trimesh.Trimesh(vertices=vertices, faces=source_handle.faces, process=False)
    target_verts:np.ndarray = applyTransfer(source, closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv)
    target_verts = target_verts.flatten()
    return target_verts

map_dict = load_maps()

def trans_task(filename:str):
    npy_name = os.path.basename(filename)
    template_name = "_".join(npy_name.split("_")[:-1])
    template_map = map_dict[template_name]

    vertices_frames = np.load(filename, allow_pickle=True)

    frames_len = vertices_frames.shape[1]
    map_frames = [template_map] * frames_len

    result = list(map(trans_vertices, vertices_frames, map_frames))
    trans_frames = np.array(result)
    target_name = filename.replace("vertices_npy", "vertices_npy_usc")
    np.save(target_name, trans_frames)

npy_files = list(glob("vocaset/vertices_npy/*.npy"))
trans_task(npy_files[0])
# with Pool(8) as p:
#     p.map(trans_task, npy_files)



# with open("vocaset/templates.pkl", 'rb') as fin:
#     template_dict = pickle.load(fin,encoding='latin1')
# for template_name, template_vertices in template_dict.items():
#     source_handle.vertices = template_vertices
#     obj_path = os.path.join("vocaset/templates", template_name + ".obj")
#     map_path = os.path.join("vocaset/trans_map", template_name + ".map")
#     source_handle.write(obj_path)
#     os.system(f"python -m zlw.contrib.zyh_topo_transfer.topology_transfer --record --map {map_path} --source {obj_path} --target data/USC_neutral.obj ")



# with open("vocaset/templates.pkl", 'rb') as fin:
#     template_dict = pickle.load(fin,encoding='latin1')
# sample:np.ndarray = list(template_dict.items())[0][1]
# trans_sample = trans_vertices(sample).reshape(14062, 3)
# target_handle = ObjHandle(vertices=trans_sample, texcoords=tex_coords, faces=faces, face_texs=face_tcs,
#                             mtl=source_handle.mtl, obj_material=source_handle.obj_material, triangulate=True)
# target_handle.write("trans_flame.obj")



# with open("vocaset/templates.pkl", 'rb') as fin:
#     template_dict = pickle.load(fin,encoding='latin1')
# trans_dict = {k: trans_vertices(v) for k, v in template_dict.items()}
# sample:np.ndarray = list(trans_dict.items())[0][1]
# sample = sample.reshape(14062, 3)
# target_handle = ObjHandle(vertices=sample, texcoords=tex_coords, faces=faces, face_texs=face_tcs,
#                             mtl=source_handle.mtl, obj_material=source_handle.obj_material, triangulate=True)
# target_handle.write("trans_template.obj")

# with open("vocaset/usc_templates.pkl", 'wb') as fout:
#     pickle.dump(template_dict, fout)

# print(template_dict)