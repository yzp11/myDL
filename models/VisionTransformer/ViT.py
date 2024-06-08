import torch
from torch import nn
from models.Transformer.mutli_head_attention import MutliHeadAttention
from models.VisionTransformer.PatchEmbedding import PatchEmbedding
from models.Transformer.mutli_head_sparse_attention import MutliHeadSparseAttention as MutliHeadAttention
#from models.Transformer.mutli_head_local_attention import MutliHeadLocalAttention as MutliHeadAttention


def init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



class Block(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 dim_feedforward,
                 dropout = 0.0):
        super(Block, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MutliHeadAttention(d_model=d_model,
                                       n_head=n_head,
                                       dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)


        self.linear1 = nn.Linear(d_model,dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward,d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2, x2, x2))

        x2 = self.dropout2(self.linear2(
            self.dropout1(self.activation(self.linear1(self.norm2(x))))))
        
        x = x + x2
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channel=3, num_classes=1000,
                 embedding_dim=768, depth=12, n_head=12, 
                 dim_feedforward=3072, dropout=0.0):
        super(VisionTransformer, self).__init__()
        
        self.num_tokens = 1

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channel, embedding_dim)
        self.num_patches = self.patch_embedding.num_patches

        self.class_token = nn.Parameter(torch.zeros(1,1,embedding_dim))
        self.position_encoding = nn.Parameter(torch.zeros(1, self.num_patches+self.num_tokens,embedding_dim))

        self.position_dropout = nn.Dropout(p=dropout)

        self.blocks = nn.Sequential(*[
            Block(d_model=embedding_dim, n_head=n_head,
                  dim_feedforward=dim_feedforward,dropout=dropout)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Linear(embedding_dim, num_classes)

        nn.init.trunc_normal_(self.position_encoding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(init_vit_weights)

    def forward(self, x):

        x = self.patch_embedding(x)

        batch_class_token = self.class_token.expand(x.shape[0],-1,-1)
        x = torch.cat((batch_class_token,x), dim=1)

        x = self.position_dropout(x + self.position_encoding)
        x = self.blocks(x)
        x = self.norm(x)

        output = x[:, 0]
        output = self.classifier(output)

        return output





def vit_base_patch16_224_in21k(num_classes: int = 21843):
    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              embedding_dim=768,
                              depth=12,
                              n_head=12,
                              num_classes=num_classes)

    return model