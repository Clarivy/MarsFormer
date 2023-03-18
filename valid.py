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
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    audio, vertice, template, one_hot = util.to_cuda(
        data['audio'],
        data['vertice'],
        data['template'],
        data['one_hot']
    )
    generated = model.predict(audio, template, one_hot).detach().cpu().numpy()

    npy_name = os.path.basename(data['data_dir']) + '.npy'
    npy_path = os.path.join(opt.results_dir, f"{opt.name}_{opt.which_epoch}", data['identity_name'])
    os.makedirs(npy_path, exist_ok=True)
    np.save(os.path.join(npy_path, npy_name), generated)

    if not opt.no_obj:
        obj_path = os.path.join(npy_path, 'obj')
        os.makedirs(obj_path, exist_ok=True)
        visualizer.frame_visualizer(generated, obj_path)
    
    print(f'Processed {data["audio_dir"]} with identity {data["identity_name"]} ')