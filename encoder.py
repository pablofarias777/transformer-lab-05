import torch
import torch.nn as nn
from attention import MultiHeadAttention
from add_norm import AddNorm
from ffn import FeedForward


class EncoderBlock(nn.Module):
    
    def __init__(self, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, h)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, src_mask=None):
       
        attn_output, _ = self.self_attention(x, x, x, mask=src_mask)
        x = self.add_norm1(x, attn_output)

        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)

        return x


class Encoder(nn.Module):
    
    def __init__(self, d_model=512, d_ff=2048, h=8, N=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, d_ff, h, dropout) for _ in range(N)]
        )

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
