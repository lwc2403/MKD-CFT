import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
from torch.autograd import Variable
from resnet import ResNet

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, dropout, max_len=5000): 
        super(PositionalEncoding, self).__init__() 
        self.dropout = nn.Dropout(p=dropout) 
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
          -(math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe) 
    def forward(self, x): 
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
        return self.dropout(x) 

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
    def forward(self, query, key, value, mask=None): 
    
        if mask is not None: 
              mask = mask.unsqueeze(1) 
        nbatches = query.size(0) 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k)
          .transpose(1, 2) for l, x in 
          zip(self.linears, (query, key, value))] 
        x, self.attn = attention(query, key, value, mask=mask, 
          dropout=self.dropout) 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) 
        return self.linears[-1](x) 

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

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) 
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size 

    def forward(self, x, mask):
        x = self.sublayer[0](x, 
          lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
                
class Transformer_m(nn.Module):
    def __init__(self, d_model, N, h):
        super(Transformer_m, self).__init__()
        self. pos_encoding = PositionalEncoding(d_model, 0.1)
        self.d_ff = d_model*4
        self.model = Encoder(
            EncoderLayer(d_model, MultiHeadAttention(h, d_model),
                         PositionwiseFeedForward(d_model, self.d_ff)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        return self.model(x, mask)
        
 
     
