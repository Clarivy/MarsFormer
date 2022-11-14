# streaming version of Faceformer

import sys
import os
import argparse

import numpy as np
import librosa

#import zlw
import time

import shutil
import keyboard
from msvcrt import getch
import pyaudio
import threading
import socket

from data_loader import load_vertices

import numpy as np
import scipy.io.wavfile as wav

import os,sys,shutil,argparse,copy,pickle
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor

import torch
import torch.nn as nn
import torch.nn.functional as F

from subprocess import call



timelist = []
index = 0
running, streaming = True, False
stream_audio = np.array([], dtype=np.int16)

# 音频录制类
class Recorder():
    def __init__(self, chunk=1024, channels=1, rate=16000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []
        self.idx = 0
        self.sleep_time = 1
        self.rec_time = 1
        #self.lines = output(initial_len=10, interval=0)
        self.count=0
    def start(self):
        threading.Thread(target=self.__recording).start()
        threading.Thread(target=self.save).start()
        print('Start!')

    def __recording(self):
        global index
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        while self._running:
            data = stream.read(self.CHUNK)
            self._frames.append(data)
            self.count+=1



        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self._running = False
        print("\n"*12)

    def save(self):
        global stream_audio
        # stream_audio0 = np.array([], dtype=np.int16)
        output=[]
        global index
        while self._running:
            if self.sleep_time >= 0:
                time.sleep(self.sleep_time)
            index = len(self._frames)
            if index != self.idx:
                frames = self._frames[self.idx: index]
                self.idx = index
                nplist = []
                for f in frames:
                    ndata = np.frombuffer(f, dtype=np.int16)
                    nplist.append(ndata)
                stream_audio = np.array(nplist).ravel()
                sleep_start = time.time()
                #single_inference(0, True, True, rec=self)
                output=(test_model())
                if(index<50):
                    ans=output
                else:
                    ans=np.concatenate((ans,output))
                print(ans.shape)
                np.save('output.npy', ans)
                start = time.time()
                # if(output.shape[0]<35 and output.shape[0]>26):
                #     s.send(output.tobytes())
                print("Waiting time: ", time.time() - start)
                self.sleep_time = self.rec_time - time.time() + sleep_start
                print("sleep_time", self.sleep_time)
            else:
                self.sleep_time = self.rec_time
        

# # 命令行参数Helper------------------------------------------------------------------------------------------
# def str2bool(val):
#     if isinstance(val, bool):
#         return val
#     elif isinstance(val, str):
#         if val.lower() in ['true', 't', 'yes', 'y']:
#             return True
#         elif val.lower() in ['false', 'f', 'no', 'n']:
#             return False
#     return False


# 主函数协程1，读取热键退出代码
def hotkey_exit():
    global running
    print("hotkey exit!")
    #session1.close()
    print('session 1 closed!')
    #session2.close()
    print('session 2 closed!')
    running = False
    getch()

# 处理结果转发
def array_send(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]



#---------------------------------------Faceformer inference---------------------------------------------------
@torch.no_grad()
def test_model(device='cuda'):
    # if not os.path.exists(args.result_path):
    #     os.makedirs(args.result_path)

    # #build model
    # model = Faceformer(args)
    # model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name))))
    # model = model.to(torch.device(args.device))
    # model.eval()

    # template_file = os.path.join(args.dataset, args.template_path)
    # with open(template_file, 'rb') as fin:
    #     templates = pickle.load(fin,encoding='latin1')

    # train_subjects_list = [i for i in args.train_subjects.split(" ")]

    # one_hot_labels = np.eye(len(train_subjects_list))
    # iter = train_subjects_list.index(args.condition)
    # one_hot = one_hot_labels[iter]
    # one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    # one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    # temp = templates[args.subject]
             
    # template = temp.reshape((-1))
    # template = np.reshape(template,(-1,template.shape[0]))
    # template = torch.FloatTensor(template).to(device=args.device)
    audio=stream_audio
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:, 0]
    wav_path = args.wav_path
    #test_name = os.path.basename(wav_path).split(".")[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    start=time.time()
    speech_array2 = audio
    #processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=device)
    print("encoder time:",time.time()-start)
    start=time.time()
    prediction = model.predict(audio_feature, template, one_hot)
    prediction = prediction.squeeze() # (seq_len, V*3) -> update to (seq_len, 55)
    prediction = prediction.cpu().numpy()
    print("prediction time:",time.time()-start)
    print("prediction shape:",prediction.shape)
    return prediction

    #np.save(os.path.join(args.result_path, test_name), prediction.detach().cpu().numpy())
#-----------------------------------------------------------------------------------------------------------

#def single_inference(index, is_recording, is_speaking, rec=None):
    

parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
parser.add_argument("--model_name", type=str, default="vocaset")
parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
parser.add_argument("--fps", type=float, default=30, help='frame rate - 30 for vocaset; 25 for BIWI')
parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
parser.add_argument("--vertice_dim", type=int, default=11793, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
parser.add_argument("--output_path", type=str, default="demo/output", help='path of the rendered video sequence')
parser.add_argument("--wav_path", type=str, default="D:\long\SHTU\\2022 Fall\MARS\FaceFormer\\test\\test.wav", help='path of the input audio signal')
parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
parser.add_argument("--condition", type=str, default="FaceTalk_170913_03279_TA", help='select a conditioning subject from train_subjects')
parser.add_argument("--subject", type=str, default="FaceTalk_170809_00138_TA", help='select a subject from test_subjects or train_subjects')
parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')
parser.add_argument("--base_model_path", type=str, required=False, default="FLAME\\",help='path of base model')
parser.add_argument("--base_template", type=str, required=False, default="mesh\\000_generic_neutral_mesh.obj", help='path of base model template')
parser.add_argument("--model_path", type=str, required=False, default="model\\100_model.pth", help='path of base pth path')
args = parser.parse_args()   



#build faceformer model
if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

#build model
model = Faceformer(args)
model.load_state_dict(torch.load(args.model_path))
from faceformer import PeriodicPositionalEncoding, init_biased_mask
model.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period, max_seq_len=6000)
model.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 6000, period=args.period)
model = model.to(torch.device(args.device))
model.eval()

template_file = os.path.join(args.dataset, args.template_path)
with open(template_file, 'rb') as fin:
    templates = pickle.load(fin,encoding='latin1')

train_subjects_list = [i for i in args.train_subjects.split(" ")]

one_hot_labels = np.eye(len(train_subjects_list))
iter = train_subjects_list.index(args.condition)
one_hot = one_hot_labels[iter]
one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
one_hot = torch.FloatTensor(one_hot).to(device=args.device)

if args.base_model_path is not None:
    temp = load_vertices(args.base_template, scale=1/100)
else:
    temp = templates[args.subject]
            
template = temp.reshape((-1))
template = np.reshape(template,(-1,template.shape[0]))
template = torch.FloatTensor(template).to(device=args.device)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#<editor-fold desc="正式流程------------------------------------------------------------------------------------------"
# 添加热键，控制随时退出
keyboard.add_hotkey('ctrl+shift+x', hotkey_exit)
# 无线传输
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect(('localhost', 55555))

while running:
    user_input = str(input("Please select operation mode from [A / X]"
                                + "\n\tPress A: start streaming and recording"
                                + "\n\tPress X: exit"
                                + "\n\t: "))
    if user_input in ['a', 'A']:
        output_type = 0
        a = input('Press a to start streaming, other keys to exit.'
                  + '\nWhen stream starts, press b to stop.'
                  + '\n: ')
        if a not in ['A', 'a']:
            break
        else:
            rec = Recorder()
            begin = time.time()
            rec.start()
            # 监听用户是否停止stream
            b = input('Press any key to stop streaming:\n')
            if b not in ['']:
                print('Stop!')
                rec.stop()
    elif user_input in ['x', 'X']:
        running = False
    else:
        print("Error: option entered not in option list.\nPlease read the guideline carefully.")
# 抹掉输出时间的10行
# print((' '*51+'\n')*10)
# sys.stdout.write("\x1b[1A"*10)
# session1.close()
# session2.close()
#</editor-fold

