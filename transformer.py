import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from utils import PositionalEncoding


class Transformer(nn.Module):
   
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, d_ff=512, h=4, N=2,
                 max_len=128, dropout=0.1):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.src_pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(d_model, d_ff, h, N, dropout)
        self.decoder = Decoder(d_model, d_ff, h, N, dropout)

        self.Wo = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
       
        src_emb = self.src_pos_enc(self.src_embedding(src))
        tgt_emb = self.tgt_pos_enc(self.tgt_embedding(tgt))

        encoder_output = self.encoder(src_emb, src_mask)

        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)

    
        logits = self.Wo(decoder_output)

        return logits

    def encode(self, src, src_mask=None):
        
        src_emb = self.src_pos_enc(self.src_embedding(src))
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        
        tgt_emb = self.tgt_pos_enc(self.tgt_embedding(tgt))
        return self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
