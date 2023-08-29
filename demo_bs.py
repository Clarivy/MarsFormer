import numpy as np
import scipy.io.wavfile as wav
import librosa
import os,sys,shutil,argparse,copy,pickle
import math,scipy
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
from data_loader import load_vertices, load_base_model
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from options.base_options import BaseOptions

import cv2
import tempfile
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # egl
import pyrender
from psbody.mesh import Mesh
import trimesh
from tqdm import tqdm

@torch.no_grad()
def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    #build model
    args.train_subjects = BaseOptions.split_subjects(args.train_subjects)
    train_subjects_list = args.train_subjects
    model = Faceformer(args)
    model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_name, f"{args.model_step}_model.pth")))

    from faceformer import PeriodicPositionalEncoding, init_biased_mask
    model.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period, max_seq_len=args.max_len)
    model.biased_mask = init_biased_mask(n_head = 4, max_seq_len = args.max_len, period=args.period)
    model = model.to(torch.device(args.device))
    model.eval()

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)
             
    speech_array, sampling_rate = librosa.load(os.path.join(args.wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("./facebook/wav2vec_processor")
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
    start=time.time()
    prediction = model.predict(audio_feature, None, one_hot)
    print("predict:"+str(time.time()-start))
    prediction = prediction.squeeze() # (seq_len, V*3)

    audio_name = os.path.basename(args.wav_path).split(".")[0]
    bs_output_file = os.path.join(args.result_path, f"{args.model_name}_{args.model_step}_{audio_name}.npy")
    vert_output_file = os.path.join(args.result_path, f"{args.model_name}_{args.model_step}_{audio_name}_vert.npy")

    print(f"Saving to {bs_output_file}")
    np.save(bs_output_file, prediction.detach().cpu().numpy())

    base_models = load_base_model(args.base_models_path, scale=0.01)
    base_model_temp = base_models[0]
    base_models = base_models[1:] - base_model_temp # - template
    base_models = base_models.reshape(55, -1)
    get_vertice = lambda x: x @ base_models + base_model_temp.reshape(-1)
    
    vert_pred = get_vertice(prediction.detach().cpu().numpy())
    print(f"Saving to {vert_output_file}")
    np.save(vert_output_file, vert_pred)
    

# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args,mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    camera_params = {'c': np.array([400, 400]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except Exception as e:
        print('pyrender: Failed rendering frame')
        print(e)
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence(args):
    wav_name = os.path.basename(args.wav_path).split(".")[0]
    template_file = args.render_template_path
    predicted_vertices_path = os.path.join(args.result_path, f"{args.model_name}_{args.model_step}_{wav_name}_vert.npy")
    video_fname = os.path.join(args.result_path, f"{args.model_name}_{args.model_step}_{wav_name}.mp4")
         
    print("rendering: ", args.wav_path)
                 
    template = Mesh(filename=template_file)
    predicted_vertices = np.load(predicted_vertices_path)
    # predicted_vertices = np.reshape(predicted_vertices,(-1,args.vertice_dim//3,3))
    predicted_vertices = np.reshape(predicted_vertices,(-1,42186//3,3))

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=args.result_path)
    
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)
    center = np.mean(predicted_vertices[0], axis=0)

    for i_frame in tqdm(range(num_frames)):
        render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(args,render_mesh, center)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()


    cmd_audio = f"ffmpeg -i {tmp_video_file.name} -i {args.wav_path} -c:v copy -c:a aac {video_fname}".split()
    call(cmd_audio)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_step", type=str, default="latest")
    parser.add_argument("--model_path", type=str, required=True, help='path of base pth path')
    parser.add_argument("--fps", type=float, default=25, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, required=False,default="M1", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')
    parser.add_argument("--base_models_path", type=str, required=False, help='path of base model')
    parser.add_argument("--no_render", action='store_true', help='whether to render the video')
    parser.add_argument("--base_only", action='store_true', help='whether to save whole model or just base vector')
    parser.add_argument('--max_len', type=int, default=600, help='number of maximum frame num')
    args = parser.parse_args()

    test_model(args)
    if not args.no_render:
        render_sequence(args)

if __name__=="__main__":
    main()
