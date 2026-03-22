import torch
import torch.nn as nn
from attention import MultiHeadAttention
from add_norm import AddNorm
from ffn import FeedForward
from utils import create_causal_mask


class DecoderBlock(nn.Module):
   
    def __init__(self, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, h)
        self.add_norm1 = AddNorm(d_model, dropout)

        self.cross_attention = MultiHeadAttention(d_model, h)
        self.add_norm2 = AddNorm(d_model, dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, y, encoder_output, tgt_mask=None, src_mask=None):
        
        if tgt_mask is None:
            seq_len = y.size(1)
            tgt_mask = create_causal_mask(seq_len, device=y.device)

        attn_output, _ = self.masked_self_attention(y, y, y, mask=tgt_mask)
        y = self.add_norm1(y, attn_output)

        cross_output, _ = self.cross_attention(
            y, encoder_output, encoder_output, mask=src_mask
        )
        y = self.add_norm2(y, cross_output)

        ffn_output = self.ffn(y)
        y = self.add_norm3(y, ffn_output)

        return y


class Decoder(nn.Module):
    
    def __init__(self, d_model=512, d_ff=2048, h=8, N=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, d_ff, h, dropout) for _ in range(N)]
        )

    def forward(self, y, encoder_output, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            y = layer(y, encoder_output, tgt_mask, src_mask)
        return y
