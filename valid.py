from options.valid_options import ValidOptions
import util.util as util
from util.visualizer import Visualizer

import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import NPFADataset
from faceformer import create_model

opt = ValidOptions().parse(save=False)

dataset = NPFADataset(opt)
visualizer = Visualizer(opt)
model = create_model(opt).cuda()

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    audio, vertice, template, one_hot = data['audio'], data['vertice'], data['template'], data['one_hot']
    audio, vertice, template, one_hot = audio.cuda(), vertice.cuda(), template.cuda(), one_hot.cuda()
    generated = model.predict(audio, template, one_hot)
    
    print(f'Processed {data["audio_dir"]} with identity {data["identity_name"]} ')