import torch
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self,  max_len, d_model,device, dropout=0.1):
        super(PositionEncoding, self).__init__()

        # Size [batch, len, dmodel]

        self.dropout = nn.Dropout(p=dropout)

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0,max_len,device=device)
        pos=pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2]= torch.sin(pos / (10000 ** (_2i/d_model)))
        self.encoding[:, 1::2]= torch.cos(pos / (10000 ** (_2i/d_model)))

    def forward(self, x):
        batch, len, dmodel = x.size()

        x = x + self.encoding[:len, :]  

        return self.dropout(x)
