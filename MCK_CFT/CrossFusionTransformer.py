import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *
import numpy as np
import math, copy, time

class CrossFusionTransformer(torch.nn.Module):
    def __init__(self, args):
        super(CrossFusionTransformer, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.args = args
        self.kernel_num = 64
        self.kernel_num_v = 64
        self.filter_sizes = [1,3,5] 
        self.filter_sizes_v = [1,3,5] 
        self.transformer = Transform_CFT(512)
        self.v_transformer = Transform_CFT(512)
        self.dropout_rate = 0.2
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders_cft = []
        self.encoders_cft_v = []
        self.encoders_cft = [  
            (setattr(self, f"encoders_cft_{i}", torch.nn.Conv1d(in_channels=512, out_channels=self.kernel_num, kernel_size=filter_size).to('cuda')),  
             getattr(self, f"encoders_cft_{i}"))  
            for i, filter_size in enumerate(self.filter_sizes)  
        ]  
        self.encoders_cft = [encoder for _, encoder in self.encoders_cft]
        self.encoders_cft_v = [  
            (setattr(self, f"encoders_cft_v_{i}", torch.nn.Conv1d(in_channels=512, out_channels=self.kernel_num, kernel_size=filter_size).to('cuda')),  
             getattr(self, f"encoders_cft_v_{i}"))  
            for i, filter_size in enumerate(self.filter_sizes)  
        ]  
        self.encoders_cft_v = [encoder for _, encoder in self.encoders_cft_v]
        
    def process_data(self,input_):
        pre_data = input_.unsqueeze(-2).view(input_.size(0), -1, self.args.sample, input_.size(2))
        pre_data = (pre_data.sum(dim=-2) / self.args.sample).squeeze(-2)  
        pre_data = self.transformer(pre_data)
        return pre_data
        
    def process_encoder(self,encoder, input_):  
        encoder_1_ = F.relu(encoder(input_.transpose(-1, -2)) )  
        kernel_size_1 = encoder_1_.size()[-1]  
        encoder_1_ = F.max_pool1d(encoder_1_, kernel_size=kernel_size_1)  
        encoder_1_ = encoder_1_.squeeze(dim=-1)  
        return encoder_1_  

    def _aggregate(self, input_1, input_2):
        encoder_outs_1 = []
        encoder_outs_2 = []
        encoder_outs_1 = [self.process_encoder(encoder, input_1) for encoder in self.encoders_cft]  
        encoding_1 = self.dropout(torch.cat(encoder_outs_1, 1))  
        output_1_pre = F.relu(encoding_1)
        encoder_outs_2 = [self.process_encoder(encoder, input_2) for encoder in self.encoders_cft_v]  
        encoding_2 = self.dropout(torch.cat(encoder_outs_2, 1))  
        output_2_pre = F.relu(encoding_2)
        output_pre = torch.cat((output_1_pre, output_2_pre), dim=1)       
        return output_pre
    def forward(self, data):
        cft_input_1 = self.process_data(data)
        cft_input_2 = self.process_data(data.transpose(-1, -2))
        output = self._aggregate(cft_input_1, cft_input_2)
        return output    

    
def attention(query, key, value, mask=None, dropout=None): 
    d_k = query.size(-1) 
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
    if mask is not None: 
        scores = scores.masked_fill(mask == 0, -1e9) 
    p_attn = F.softmax(scores, dim = -1) 
    if dropout is not None: 
        p_attn = dropout(p_attn) 
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def get_rel_pos(self, x):
        return max(self.k*-1, min(self.k, x))
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Transform_CFT(nn.Module):
    def __init__(self, d_model):
        super(Transform_CFT, self).__init__()
        self.filters = [1, 3, 5]
        self.layers = 2
        self.heads = 8
        self.model = Encoder(EncoderLayer(d_model, MultiHeadAttention(self.heads, d_model),PositionwiseFeedForward(d_model, d_model*4), 0.1),self.layers)
    def forward(self, input_data, mask=None):
        return self.model(input_data, mask)  

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))




