import torch
import torch.nn as nn


class AddNorm(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        
        return self.norm(x + self.dropout(sublayer_output))
