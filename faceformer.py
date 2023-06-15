import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model
from whisper import WhisperModel
from data_loader import load_base_model, load_vertices
import os

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1), period, rounding_mode='floor')
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(T, S):
    mask = torch.ones(T, S)
    mask.fill_diagonal_(0)
    return (mask==1).cuda()

def get_base_dec_func(base_models):
    def transformer(base_vec):
        return torch.matmul(base_vec, base_models)
    return transformer

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Faceformer(nn.Module):
    def __init__(self, opt):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.base_models = None
        if hasattr(opt, 'facial_mask') and opt.facial_mask is not None:
            self.facial_mask = opt.facial_mask
            self.nonfacial_mask = opt.nonfacial_mask
        else:
            self.facial_mask = None
        self.vertice_dim = opt.vertice_dim

        self.audio_encoder = Wav2Vec2Model.from_pretrained("./facebook/wav2vec")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, opt.feature_dim)
        # motion encoder
        self.vertice_map = nn.Linear(self.vertice_dim, opt.feature_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(opt.feature_dim, period = opt.period, max_seq_len=opt.max_len)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = opt.max_len, period=opt.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=opt.feature_dim, nhead=4, dim_feedforward=2*opt.feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # motion decoder
        self.vertice_map_r = nn.Linear(opt.feature_dim, self.vertice_dim)
        
        # encoding the template
        self.obj_vector = nn.Linear(len(opt.train_subjects), opt.feature_dim, bias=False)
        self.activation_func = nn.LeakyReLU(negative_slope=opt.neg_penalty)

        self.max_lex = opt.max_len
        
        # TODO: Wether initialize the base_map_r
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio, vertice, template, one_hot, criterion, teacher_forcing=False):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        if template is not None:
            template = template.unsqueeze(1) # (1,1, V*3)
        frame_num = vertice.shape[1]
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio, frame_num=frame_num).last_hidden_state
        hidden_states = self.audio_feature_map(hidden_states)

        negative_penalty = torch.tensor(0.).cuda()

        if teacher_forcing:
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb  
            if template is not None:
                vertice_input = torch.cat((template,vertice[:,:-1]), 1) # shift one position
                vertice_input = vertice_input - template
                vertice_input = self.vertice_map(vertice_input)
                vertice_input = vertice_input + style_emb
                vertice_input = self.PPE(vertice_input)
            else:
                vertice_input = self.vertice_map(vertice)
                vertice_input = vertice_input + style_emb
                vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().cuda()
            memory_mask = enc_dec_mask(vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
        else:
            for i in range(frame_num):
                if i==0:
                    vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().cuda()
                memory_mask = enc_dec_mask(vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        # When predicting motion
        if template is not None:
            vertice_out = vertice_out + template
            vertice_out = self.get_facial_area(vertice_out)
            vertice = self.get_facial_area(vertice)

        loss = criterion(vertice_out, vertice) # (batch, seq_len, V*3)
        
        total_loss = torch.mean(loss) - negative_penalty
        losses = {
            'total_loss': total_loss,
            'negative_penalty': negative_penalty,
        }
        return losses

    def predict(self, audio, template, one_hot):
        max_len = self.max_lex
        if template is not None:
            template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio).last_hidden_state
        frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)
        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            
            if i < max_len-1:
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().cuda()
                memory_mask = enc_dec_mask(vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                vertice_out_all = copy.deepcopy(vertice_out)
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)
            else:
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().cuda()
                memory_mask = enc_dec_mask(vertice_input.shape[1], hidden_states.shape[1]-(i-max_len+1))
                vertice_out = self.transformer_decoder(vertice_input, hidden_states[:,i-max_len+1:,:], tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                vertice_out_all = torch.cat((vertice_out_all, vertice_out[:,-1,:].unsqueeze(1)), 1)
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb[:,1:,:], new_output), 1)

        if template is not None:
            vertice_out_all = vertice_out_all + template
        return vertice_out_all
    
    def get_facial_area(self, vertice: torch.Tensor):
        if self.facial_mask == None:
            return vertice
        return vertice.view(1, -1, 3)[:, self.facial_mask, :].flatten(1)
        
        

def create_model(opt):
    model = Faceformer(opt)

    if (not opt.isTrain) or (opt.continue_train):
        pretrained_path = os.path.join(opt.checkpoints_dir, opt.name)
        model_path = os.path.join(pretrained_path, f'{opt.which_epoch}_model.pth')
        model.load_state_dict(torch.load(model_path))
        print(f"Model [{type(model).__name__}] is loaded from {model_path}")
    
    print(f"Model [{type(model).__name__}] is created")

    return model