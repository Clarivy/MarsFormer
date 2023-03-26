from options.valid_options import ValidOptions
import util.util as util
from util.visualizer import Visualizer

import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataset
from faceformer import create_model

opt = ValidOptions().parse(save=False)

dataset = get_dataset(opt)
visualizer = Visualizer(opt)
model = create_model(opt).cuda()

# test
for i, total_data in enumerate(dataset):
    if i >= opt.how_many:
        break
    audio, vertice, template, one_hot = util.to_cuda(
        total_data['audio'],
        total_data['vertice'],
        total_data['template'],
        total_data['one_hot']
    )
    generated = model.predict(audio, template, one_hot).detach().cpu().numpy()

    npy_name = os.path.basename(total_data['data_dir'][0]) + '.npy'
    npy_path = os.path.join(opt.results_dir, f"{opt.name}_{opt.which_epoch}", total_data['identity_name'][0])
    os.makedirs(npy_path, exist_ok=True)
    np.save(os.path.join(npy_path, npy_name), generated)

    if not opt.no_obj:
        obj_path = os.path.join(npy_path, 'obj')
        os.makedirs(obj_path, exist_ok=True)
        visualizer.frame_visualizer(generated, obj_path)
    
    print(f'Processed {total_data["audio_dir"][0]} with identity {total_data["identity_name"][0]} ')