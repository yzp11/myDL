import torch
from torch import nn

class PatchEmbedding(nn.Module):

    def __init__(self, image_size=224, patch_size=16, in_channel=3, embedding_dim=768):

        super(PatchEmbedding, self).__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.conv = nn.Conv2d(in_channel, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm([self.num_patches,embedding_dim])

    def forward(self, x):
        # Input[batch, channel, height, width]
        # Output[batch, num_patches, emdedding_dim]
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        x = self.conv(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x