import torch
from torch import nn
import math

class DotProductSparseAttention(nn.Module):

    def __init__(self, dropout= 0.0):
        super(DotProductSparseAttention,self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None, num_k=170):
        #Size [batch, head, len, d_tensor]

        batch, head, len, d_tensor = k.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask ==0, -1e9)

        if  num_k>k.size()[2]:
            num_k=k.size()[2]

        if  num_k:
            va,_ = torch.topk(score, num_k)

            vk=va[:,:,:,-1].unsqueeze(3).expand_as(score)
            mask_k=torch.lt(score,vk)
            score=score.masked_fill(mask_k,-1e18)

        score = self.softmax(score)
        score = self.dropout(score)

        attn_output = score @ v

        return attn_output

class MutliHeadSparseAttention(nn.Module):

    def __init__(self, d_model, n_head, dropout=0.0):
        super(MutliHeadSparseAttention, self).__init__()
        self.n_head=n_head
        self.attention = DotProductSparseAttention(dropout=dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v= self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)

        out = self.attention(q, k, v, mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out
    
    def split(self, tensor):
        
        #Input[batch, len, d_model]
        #Output[batch, head, len, d_tensor]

        batch, len, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch, len, self.n_head, d_tensor).transpose(1, 2)

        return tensor
    
    def concat(self, tensor):

        #Input[batch, head, len, d_tensor]
        #Output[batch, len, d_model]

        batch, head, len, d_tensor = tensor.size()
        d_model = d_tensor * head

        tensor = tensor.transpose(1,2).contiguous().view(batch,len,d_model)

        return tensor