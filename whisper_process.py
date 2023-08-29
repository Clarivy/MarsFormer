import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from transformers import WhisperModel,WhisperConfig
from transformers import WhisperFeatureExtractor,WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple
from glob import glob
import tqdm
import librosa
import os
_CONFIG_FOR_DOC = "WhisperConfig"


# linear interpolation layer
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)

class Whisper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = WhisperModel.from_pretrained("openai/whisper-large-v2")
        for param in self.model.parameters():
            param.requires_grad = False
        self.requires_grad_ = False
        self.decoder_input_ids = (torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id).cuda()

        # self.encoder = self.model.get_encoder()
        # self.model.freeze_encoder()
    def forward(
        self,
        input_features
    ):
        #[1,1500,1280]
        hidden_states = self.model(input_features, decoder_input_ids=self.decoder_input_ids).last_hidden_state.cuda()
        # hidden_states = hidden_states.reshape(1, -1, 64)
        # input_values  [1, 86667]       [1, 112299]
        # hidden_states = self.feature_extractor(input_values)
        # hidden_states [1, 512, 270]    [1, 512, 350]
        # hidden_states = hidden_states.transpose(1, 2)
        # hidden_states [1, 270, 512]    [1, 350, 512]
        # hidden_states = linear_interpolation(hidden_states, 50, 30,output_len=frame_num)
        # hidden_states [1, 163, 768]    [1, 210, 512]

        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=hidden_states,
        )
audio_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
audio_encoder = Whisper().cuda()

target_files = sorted(glob("/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/GNPFA_CREMA-D/train/*NEU*/audio.wav"))
for wav_data in tqdm.tqdm(target_files):
    print(f"Processing {wav_data}")
    speech_array, sampling_rate = librosa.load(wav_data, sr=16000)
    processed_audio = torch.FloatTensor(audio_processor(speech_array,sampling_rate=16000,return_tensors="pt").input_features).cuda()
    encoded_audio = audio_encoder(processed_audio).last_hidden_state
    output_file = os.path.join(os.path.dirname(wav_data), "audio_whisper.npy")
    print(f"Saving to {output_file}")
    # print(encoded_audio.shape)
    np.save(output_file, encoded_audio.cpu().detach().numpy())