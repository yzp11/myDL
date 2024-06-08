from torch import nn
import torch

from models.Transformer.mutli_head_attention import MutliHeadAttention
from models.Transformer.position_encoding import PositionEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MutliHeadAttention(d_model,nhead,dropout)

        self.linear1 = nn.Linear(d_model,dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward,d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, src, src_mask = None):
        src2 = self.self_attn(src, src, src, src_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MutliHeadAttention(d_model, nhead, dropout)
        self.mutlihead_attn = MutliHeadAttention(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, dec, enc, trg_mask, src_mask):
        dec2 = self.self_attn(dec, dec, dec, trg_mask)

        dec = dec + self.dropout1(dec2)
        dec = self.norm1(dec)

        if enc is not None:
            dec2 = self.mutlihead_attn(dec, enc, enc, src_mask)

            dec = dec + self.dropout2(dec2)
            dec = self.norm2(dec)

        dec2 = self.linear2(self.dropout(self.activation(self.linear1(dec))))

        dec = dec + self.dropout3(dec2)
        dec = self.norm3(dec)
        return dec
    

class Encoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, dropout, n_layers, device, max_len):
        super(self, Encoder).__init__()

        self.position_encoding = PositionEncoding(max_len=max_len,
                                                  d_model=d_model,
                                                  device=device,
                                                  dropout=dropout)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout)
                                     for _ in range(n_layers)])
        
    def forward(self, src, src_mask):

        src = self.position_encoding(src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src
    

class Decoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, dropout, n_layers, device, max_len):
        super(self, Encoder).__init__()

        self.position_encoding = PositionEncoding(max_len=max_len,
                                                  d_model=d_model,
                                                  device=device,
                                                  dropout=dropout)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout)
                                     for _ in range(n_layers)])
        
    def forward(self, trg, src, trg_mask, src_mask):

        trg = self.position_encoding(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        return trg
    
class Transformer(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, dropout, n_layers, device, max_len,num_class):
        super().__init__()
        
        self.classifier = nn.Linear(d_model, num_class)
        self.encoder = Encoder(d_model=d_model,
                               dim_feedforward=dim_feedforward,
                               nhead=nhead,
                               dropout=dropout,
                               n_layers=n_layers,
                               device=device,
                               max_len=max_len)
        
        self.decoder = Decoder(d_model=d_model,
                               dim_feedforward=dim_feedforward,
                               nhead=nhead,
                               dropout=dropout,
                               n_layers=n_layers,
                               device=device,
                               max_len=max_len)


    def forward(self, src, trg, src_mask, trg_mask):
        src = self.encoder(src, src_mask)
        output = self.decoder(trg, src, trg_mask, src_mask)
        output = self.classifier(output)
        return output