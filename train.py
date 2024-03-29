import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from options.train_options import TrainOptions
from data_loader import get_dataset
from faceformer import create_model
import util.util as util
from util.visualizer import Visualizer
from collections import Counter

util.enable_reproducibility() # Make reprducibility

opt = TrainOptions().parse()
model_save_dir = os.path.join(opt.checkpoints_dir, opt.name)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1

train_dataset, test_dataset = get_dataset(opt)
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
total_size = train_dataset_size + test_dataset_size
print('#dataset contains %d video' % total_size)
print('#train_dataset: %d ' % train_dataset_size)
print('#test_dataset: %d ' % test_dataset_size)

total_steps = (start_epoch - 1) * train_dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq


model = create_model(opt).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=opt.lr)
visualizer = Visualizer(opt)

for epoch in range(start_epoch, opt.epoch_num + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % train_dataset_size

    # train
    model.train()
    for i, total_data in enumerate(train_dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += 1
        epoch_iter += 1
        # teacher forcing
        teacher_forcing = False
        if opt.teacher_forcing:
            teacher_forcing_ratio = max(0.8 - epoch / opt.epoch_num, 0)
            if(torch.rand(1) < teacher_forcing_ratio):
                teacher_forcing = True
            else:
                teacher_forcing = False


        # collect input data from data loader
        data = util.to_cuda(total_data)
        audio, vertice, template, one_hot = (data.get(key) for key in [
            "audio",
            "vertice",
            "template",
            "one_hot"
        ])

        ############## Forward Pass ######################
        losses = model(audio, vertice, template, one_hot, criterion, teacher_forcing)

        ############### Backward Pass ####################
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()


        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in losses.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            util.save_model(model, model_save_dir, 'latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= train_dataset_size:
            break
    
    # valid
    model.eval()
    with torch.no_grad():
        total_errors = Counter()
        for i, total_data in enumerate(test_dataset):

            # collect input data from data loader
            data = util.to_cuda(total_data)
            audio, vertice, template, one_hot = (data.get(key) for key in [
                "audio",
                "vertice",
                "template",
                "one_hot"
            ])

            ############## Forward Pass ######################
            #No teacher forcing in eval
            losses = model(audio, vertice, template, one_hot, criterion)

            ############## Display results and errors ##########
            ### print out errors
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in losses.items()}
            total_errors = total_errors + Counter(errors)
        average_errors = {k: v / len(test_dataset) for k, v in total_errors.items()}
        visualizer.plot_valid_errors(average_errors, epoch)
        visualizer.print_valid_errors(average_errors, epoch)
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.epoch_num, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        util.save_model(model, model_save_dir, 'latest')
        util.save_model(model, model_save_dir, str(epoch))
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
    
    ### valid