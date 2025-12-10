"""
This script is directly adapted from https://github.com/Zch0414/hlip/blob/master/src/hlip/patch_embed.py
"""

import torch
import torch.nn as nn
from timm.models._manipulate import checkpoint


class ConvStem3D(nn.Module):
    """ 
    3D version of ConvStem,
    from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=[48, 224, 224], patch_size=[16, 16, 16], in_chans=1, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()
        assert len(list(img_size)) == 3, 'Specify the input size at every dimension'
        assert len(list(patch_size)) == 3, 'Specify the patch size at every dimension'

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        _rate_z, _rate_y, _rate_x = 1, 1, 1
        for _ in range(4):
            _s_z = 2 if _rate_z < patch_size[0] else 1
            _s_y = 2 if _rate_y < patch_size[1] else 1
            _s_x = 2 if _rate_x < patch_size[2] else 1
            stem.append(nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=(_s_z, _s_y, _s_x), padding=(1, 1, 1), bias=False))
            stem.append(nn.BatchNorm3d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2; _rate_z *= 2; _rate_y *= 2; _rate_x *= 2

        stem.append(nn.Conv3d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.grad_checkpointing = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, x):
        B, N, C, D, H, W = x.shape
        x = x.view(-1, C, D, H, W) # BN * C * D * H * W
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(self.proj, x)
        else:
            x = self.proj(x)

        # BN * C' * D' * H' * W' -> B * N * D' * H' * W' * C'
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, N, self.grid_size[0], self.grid_size[1], self.grid_size[2], -1) 
        x = self.norm(x)
        return x
    

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=[48, 224, 224], patch_size=[16, 16, 16], in_chans=1, embed_dim=384, norm_layer=None, **kwargs):
        super().__init__()
        assert len(list(img_size)) == 3, 'Specify the input size at every dimension'
        assert len(list(patch_size)) == 3, 'Specify the patch size at every dimension'

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=kwargs["bias"])
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, N, C, D, H, W = x.shape
        x = x.view(-1, C, D, H, W) # BN * C * D * H * W
        x = self.proj(x)

        # BN * C' * D' * H' * W' -> B * N * D' * H' * W' * C'
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, N, self.grid_size[0], self.grid_size[1], self.grid_size[2], -1) 
        x = self.norm(x)
        return x