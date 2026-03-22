import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(q, k, v, mask=None):
   
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.matmul(attn_weights, v)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
  
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0, "d_model deve ser divisível por h"
        self.h = h
        self.d_k = d_model // h


        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)

        q = self.W_q(q).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch, -1, self.h, self.d_k).transpose(1, 2)

        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)

        output = self.W_o(attn_output)
        return output, attn_weights
