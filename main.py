import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders
from faceformer import Faceformer

from tensorboardX import SummaryWriter

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, writer:SummaryWriter,epoch=100):
    save_path = os.path.join(args.logdir, "save", args.name)
    if os.path.exists(save_path):
        os.rmdir(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    for e in range(epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            loss = model(audio, template,  vertice, one_hot, criterion,teacher_forcing=False)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))
        # validation
        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all,file_name in dev_loader:
            # to gpu
            audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                loss = model(audio, template,  vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
                writer.add_scalar("loss/train", loss.item(), global_step=iteration)
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:,iter,:]
                    loss = model(audio, template,  vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())
                    writer.add_scalar("loss/val", loss.item(), global_step=iteration)
                        
        current_loss = np.mean(valid_loss_log)
        writer.add_scalar("loss/average", current_loss, global_step=e)
        
        if (e > 0 and e % 5 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))
            print("Saved model at epcoh: {}".format(e+1))

        print("epcoh: {}, current loss:{:.7f}".format(e+1,current_loss))    
    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    result_path = os.path.join(args.dataset,args.result_path)
    if os.path.exists(result_path):
        os.rmdir(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.logdir, "save", args.name)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()
   
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument("--load_model", type=str, default="", help='path of pretrained model')
    parser.add_argument("--base_model_path", type=str, required=False, help='path of base model')
    parser.add_argument("--base_template", type=str, required=False, help='path of base model template')
    parser.add_argument("--logdir", type=str, default="checkpoints",required=False, help='path of tensorboard logger')
    parser.add_argument("--name", type=str,required=True, help='name to save_model')
    parser.add_argument("--neg_penalty", type=float,required=False, default=1e-7, help='penalty for negative value in the base vector')
    args = parser.parse_args()

    # Tensorboard logger
    logdir = os.path.join(args.logdir, "web", args.name)

    if os.path.exists(logdir):
        print(f"logdir {logdir} exists, remove it / Ingore / exit [Y/I/n]")
        while True:
            ans = input().strip().lower()
            if ans == "y":
                shutil.rmtree(logdir)
                os.makedirs(logdir)
                break
            elif ans == "i":
                break
            elif ans == "n":
                exit()
    else:
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)


    #build model
    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))

    if args.load_model != "":
        model.load_state_dict(torch.load(args.load_model)) 

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    dataset = get_dataloaders(args)

    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], dataset["valid"],model, optimizer, criterion, writer, epoch=args.max_epoch)
    
    test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()