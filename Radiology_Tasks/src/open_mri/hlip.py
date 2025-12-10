"""
This code was adapted from an older version of HLIP's implementation 
"""

import sys
sys.path.append('.')

from functools import partial

import torch
import torch.nn as nn

from timm.models import register_model, build_model_with_cfg, checkpoint
from timm.models.vision_transformer import VisionTransformer

from .patch_embed import PatchEmbed3D, ConvStem3D
from .pos_embed import mri_pos_embed, resample_series_posemb, resample_spatial_posemb

# This is an implementation of HLIP
class HLIP(VisionTransformer):
    def __init__(self, **kwargs):
        self.max_num_series = kwargs.pop('max_num_series', 40)
        self.study_attn_indexes = kwargs.pop('study_attn_indexes', [2, 5, 8, 11])
        self.useseriename = kwargs.pop('use_serienames',False)
        self.is_itemizedclip = kwargs.pop('is_itemizedclip',False)
        if self.useseriename:
            SNEncoderLoc = kwargs.pop('pretrained_series_name_encoder',None)

        

        super().__init__(**kwargs)


        if self.useseriename: # use sequence name as sequence embedding, instead of custom-trained sequence embedding
            sys.path.append('../open_mri/prima1')
            self.serienameencoder = torch.load(SNEncoderLoc,map_location='cpu',weights_only=False)
            self.serienameencoder.prelinear = True
            delattr(self.serienameencoder,'linear')
            self.sne_proj = torch.nn.Linear(self.serienameencoder.embsize,self.embed_dim)

        # reset pos_embed
        spatial_posemb, series_posemb = mri_pos_embed(
            max_num_series=self.max_num_series,
            grid_size=self.patch_embed.grid_size,
            embed_dim=self.embed_dim,
            pretrained_posemb=None,
        )
        if not self.useseriename:
            self.series_posemb = nn.Parameter(series_posemb)
            self.series_posemb.requires_grad = False
        self.spatial_posemb = nn.Parameter(spatial_posemb)
        self.spatial_posemb.requires_grad = False
    # use sequence name encoder to get sequence names. Sequence and series are used interchangeably here.
    def get_serie_name_encodings(self,serienames):
        bs,serienum,seqlen = serienames.shape
        if self.grad_checkpointing and not torch.jit.is_scripting():
            outs = checkpoint(self.serienameencoder,serienames.view(bs*serienum,seqlen))
            outs = checkpoint(self.sne_proj,outs)
        else:
            outs = self.serienameencoder(serienames.view(bs*serienum,seqlen))
            outs = self.sne_proj(outs)
        return outs.view(bs,serienum,self.embed_dim)

    def _pos_embed(self, x, serienames = None):
        # x: [bs, num_series, d, h, w, c]
        bs, num_series, d, h, w, _ = x.shape
        spatial_posemb = resample_spatial_posemb(self.spatial_posemb, (d, h, w), self.patch_embed.grid_size)
        
        if self.useseriename:
            series_posemb = self.get_serie_name_encodings(serienames)
            pos_embed = spatial_posemb[:, None, :, :, :, :]
            pos_embed = pos_embed.expand(bs, -1, -1, -1, -1, -1)
            pos_embed = pos_embed + series_posemb[:,:,None,None,None,:]
        else:
            series_posemb = resample_series_posemb(self.series_posemb, num_series, is_train = bs!=1)
            pos_embed = series_posemb[:, :, None, None, None, :] + spatial_posemb[:, None, :, :, :, :]
            pos_embed = pos_embed.expand(bs, -1, -1, -1, -1, -1)

        pos_embed = pos_embed.flatten(2, 4).flatten(0, 1) # [bs * num_series, d * h * w, c]
        x = x.flatten(2, 4).flatten(0, 1) # [bs * num_series, d * h * w, c]
        
        x = self.pos_drop(x + pos_embed)

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)

        return x, num_series
    
    def _series_partition(self, x, num_series):
        """
        Partition a study into non-overlapping series.
        Args:
            x (tensor): input tokens with [B, num_prefix_tokens + num_series * L, C].
            num_series (int): number of series in one study.

        Returns:
            x: x after partition with [B * num_series, num_prefix_tokens + L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        B, NL, C = src.shape

        prefix_tokens = prefix_tokens.view(B, 1, self.num_prefix_tokens, C).expand(-1, num_series, -1, -1).contiguous()
        src = src.view(B, num_series, NL//num_series, C)
        
        x = torch.cat([prefix_tokens, src], dim=2)
        x = x.view(-1, self.num_prefix_tokens+NL//num_series, C)
        return x


    def _series_unpartition(self, x, num_series):
        """
        series unpartition into original study.
        Args:
            x (tensor): input tokens with [B * num_series, num_prefix_tokens + L, C].
            num_series (int): number of series in one study.

        Returns:
            x: unpartitioned series with [B, num_prefix_tokens + num_series * L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        NB, L, C = src.shape

        prefix_tokens = prefix_tokens.view(NB//num_series, num_series, self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(NB//num_series, num_series, L, C).view(NB//num_series, num_series * L, C)

        x = torch.cat([prefix_tokens, src], dim=1)
        return x
        
    def forward_features(self, x, serienames = None):
        x = self.patch_embed(x) # starts from partition status: [b, num_series, d, h, w, c]
        x, num_series = self._pos_embed(x, serienames = serienames)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx, blk in enumerate(self.blocks):
            if idx in self.study_attn_indexes and idx-1 not in self.study_attn_indexes:
                x = self._series_unpartition(x, num_series)
            
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
            
            if idx in self.study_attn_indexes and idx+1 not in self.study_attn_indexes and idx+1 < len(self.blocks):
                x = self._series_partition(x, num_series) 

        return self.norm(x)
    
    def forward(self, x):
        serienames = None
        if type(x) is tuple:
            if len(x[1].size()) > 1:
                x,serienames = x
            else:
                x,_ = x
        x = self.forward_features(x, serienames = serienames)
        if self.is_itemizedclip:
            return torch.cat([self.forward_head(x).unsqueeze(1),x [:, self.num_prefix_tokens:, :]],dim=1)
        x = self.forward_head(x)
        return x


def custom_checkpoint_filter_fn(state_dict, model, patch_size=[16, 16, 16]):
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)

    # determine whether the cls_token has corresponding pos_embed
    embed_len = state_dict['pos_embed'].shape[1]
    if torch.sqrt(torch.tensor(embed_len)) != torch.sqrt(torch.tensor(embed_len)).floor():
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:]

    for k, v in state_dict.items():
        if 'patch_embed' in k:
            if model.patch_embed.__class__ == PatchEmbed3D:
                if 'weight' in k:
                    if (v.shape[2], v.shape[3]) != (patch_size[1], patch_size[2]):
                        v = torch.nn.functional.interpolate(v, size=(patch_size[1], patch_size[2]), mode='bicubic')
                    v = v.sum(dim=1, keepdim=True).unsqueeze(2).repeat(1, 1, patch_size[0], 1, 1).div(patch_size[0])
            else:
                continue
        if 'pos_embed' in k:
            spatial_posemb, series_posemb = mri_pos_embed(
                max_num_series = model.max_num_series,
                grid_size = model.patch_embed.grid_size,
                embed_dim = model.embed_dim,
                pretrained_posemb = v
            )
            out_dict['spatial_posemb'] = spatial_posemb
            out_dict['series_posemb'] = series_posemb
            continue
        out_dict[k] = v
    return out_dict


def custom_create_vision_transformer(variant, **kwargs):
    kwargs.pop('pretrained_cfg_overlay')
    return build_model_with_cfg(
        model_cls=HLIP,
        variant=variant,
        pretrained_cfg_overlay=dict(first_conv=None),
        pretrained_strict=False,
        pretrained_filter_fn=partial(custom_checkpoint_filter_fn, patch_size=kwargs['patch_size']),
        **kwargs,
    )

@register_model
def hlip_base_seriename_81616_itemizedclip(pretrained=True, **kwargs):
    model_args = dict(
        study_attn_indexes=[2, 5, 8, 11], max_num_series=40,
        img_size=[48, 224, 224], patch_size=[8, 16, 16],
        in_chans=1, depth = 12, embed_dim=768, num_heads=12, num_classes=0, no_embed_class=True, pos_embed='none',
        drop_path_rate=0.2, use_serienames = True, pretrained_series_name_encoder = '../15t.pt',
        embed_layer=PatchEmbed3D, is_itemizedclip=True
    )
    model = custom_create_vision_transformer('vit_base_patch16_224.mae', pretrained=pretrained, **dict(model_args, **kwargs))
    return model



