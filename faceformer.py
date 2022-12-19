import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model
from data_loader import load_base_model, load_vertices
import performer_pytorch
from nystrom_attention import NystromAttention
from linformer_pytorch import MHAttention
from linear_attention_transformer import LinearAttentionTransformer
import slim_performer_model

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
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
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
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

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

import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from typing import Optional, Tuple

class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
            #attn_output, attn_output_weights = performer_pytorch.FastAttention(query, key, value)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
            #attn_output, attn_output_weights = performer_pytorch.FastAttention(query, key, value)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)

        self.self_attn = performer_pytorch.SelfAttention(dim=d_model, heads=nhead,dim_head=d_model//nhead, causal = True,dropout=dropout).cuda()
        self.multihead_attn = performer_pytorch.CrossAttention(dim=d_model, heads=nhead,dim_head=d_model//nhead, causal = False,dropout=dropout).cuda()
        
        # self.self_attn = performer_pytorch.FastAttention(nb_features=d_model, dim_heads=nhead, causal = True)
        # self.multihead_attn = performer_pytorch.FastAttention(nb_features=d_model, dim_heads=nhead, causal = False)
        # self.self_attn = slim_performer_model.MultiHeadAttention(feature_type='favor+',n_heads=nhead,hidden_dim=d_model,compute_type='iter')
        # self.multihead_attn = slim_performer_model.MultiHeadAttention(feature_type='favor+',n_heads=nhead,hidden_dim=d_model,compute_type='ps_parallel')
        # self.self_attn = MHAttention(
        #     dim_k=d_model, # What to sample the input length down to
        #     nhead=nhead, # Number of heads
        #     dropout=dropout, # Dropout for each of the heads
        # )
        # self.self_attn = MHAttention(
        #     input_size=d_model, # Dimension 1 of the input
        #     channels=d_model, # Dimensidon 2 of the input
        #     dim=d_model, # Dim of each attn head
        #     dim_k=d_model, # What to sample the input length down to
        #     nhead=nhead, # Number of heads
        #     dropout=dropout, # Dropout for each of the heads
        # )
        # self.self_attn = LinearAttentionTransformer(
        #     dim = d_model,
        #     heads = nhead,
        #     depth = 1,
        #     max_seq_len = 8192,
        #     n_local_attn_heads = 4,
        #     causal=True,
        # ).cuda()
        # self.multihead_attn = LinearAttentionTransformer(
        #     dim = d_model,
        #     heads = nhead,
        #     depth = 1,
        #     max_seq_len = 8192,
        #     n_local_attn_heads = 4
        # ).cuda()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        tgt2 = self.self_attn(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.multihead_attn(x=tgt, context=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))




def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.base_models = None

        self.dataset = args.dataset
        self.audio_encoder = Wav2Vec2Model.from_pretrained("./facebook/wav2vec")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        decoder_layer = TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        #self.performer_decoder = performer_pytorch.Performer(dim=args.feature_dim, heads=4, dim_head=args.feature_dim//4, causal=True, depth=1).cuda()
        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        
        # style embedding
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.device = args.device
        

        # base 
        if args.base_model_path is not None:
            self.base_template = torch.tensor(load_vertices(args.base_template, scale=1./100), dtype=torch.float)
            self.base_models = torch.tensor(load_base_model(args.base_model_path, scale=1./100), dtype=torch.float) - self.base_template
            self.base_models = self.base_models.reshape(self.base_models.size(0), -1)
            self.device_base_models = self.base_models.clone().to(self.device)
            self.base_map_r = nn.Linear(args.feature_dim, self.base_models.shape[0])
            self.activation_func = nn.LeakyReLU(negative_slope=args.neg_penalty)
            # self.activation_func = nn.Tanh()

        
        # TODO: Wether initialize the base_map_r
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)


    def forward(self, audio, template, vertice, one_hot, criterion, writer = None, global_step=None,teacher_forcing=True):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        frame_num = vertice.shape[1]
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        if self.dataset == "BIWI":
            if hidden_states.shape[1]<frame_num*2:
                vertice = vertice[:, :hidden_states.shape[1]//2]
                frame_num = hidden_states.shape[1]//2
        hidden_states = self.audio_feature_map(hidden_states)

        negative_penalty = torch.tensor(0., device=self.device)

        if teacher_forcing:
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb  
            vertice_input = torch.cat((template,vertice[:,:-1]), 1) # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            #vertice_out = self.performer_decoder(vertice_input, hidden_states)
            if self.base_models is not None:
                vertice_out = self.base_map_r(vertice_out)
                vertice_out = self.activation_func(vertice_out)
                negative_penalty = negative_penalty + torch.sum(vertice_out[vertice_out<0])
                # print("Matmuling |", vertice_out.shape, self.base_models.shape)
                vertice_out = vertice_out @ self.device_base_models
            else:
                vertice_out = self.vertice_map_r(vertice_out)
        else:
            for i in range(frame_num):
                if i==0:
                    vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                #vertice_out = self.performer_decoder(vertice_input, hidden_states)
                if self.base_models is not None:
                    vertice_out = self.base_map_r(vertice_out)
                    vertice_out = self.activation_func(vertice_out)
                    negative_penalty = negative_penalty + torch.sum(vertice_out[vertice_out<0])
                    # print("Matmuling |", vertice_out.shape, self.base_models.shape)
                    vertice_out = vertice_out @ self.device_base_models
                else:
                    vertice_out = self.vertice_map_r(vertice_out)
                
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        # print(vertice_out.shape) # 1 * n * 15069
        if self.base_models is not None:
            # vertice_out is (batch, seq_len, V*3) 
            loss = criterion(vertice_out, vertice) # penalty for negative base
        else:
            loss = criterion(vertice_out, vertice) # (batch, seq_len, V*3)
        if writer is not None:
            writer.add_scalar("loss/deviation", torch.mean(loss).item(), global_step=global_step)
            writer.add_scalar("loss/negative_penalty", negative_penalty.item(), global_step=global_step)
        loss = torch.mean(loss) - negative_penalty
        return loss

    def predict(self, audio, template, one_hot, base_only = False):
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)
        base_vec_arr = []

        output_frame = []
        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            if self.base_models is not None:
                vertice_out = self.base_map_r(vertice_out)
                vertice_out = self.activation_func(vertice_out)
                vertice_out = vertice_out.clip(0, 1)
                # print("Matmuling |", vertice_out.shape, self.base_models.shape)
                base_vec_arr = vertice_out
                vertice_out = vertice_out @ self.device_base_models
            else:
                vertice_out = self.vertice_map_r(vertice_out)
            # if i == 0:
            #     output_frame = vertice_out
            # else:
            #     print(vertice_out.shape, vertice_out[:,-1:,:].shape)
            #     print(output_frame.shape)
            #     output_frame = torch.cat((output_frame, vertice_out[:,-1:,:]), 1)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)
            # print(vertice_emb.shape)
            # if vertice_emb.size(1) > 600:
            #     start = vertice_emb.size(1) - 600
            #     vertice_emb = vertice_emb[:, start :, :]

        # output_frame = output_frame + template
        if base_only:
            return base_vec_arr
        else:
            vertice_out = vertice_out + template
            return vertice_out
