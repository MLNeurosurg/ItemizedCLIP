"""
Implementation of ItemizedCLIP model adapted from FLAIR model
"""

from open_clip.model import CustomTextCLIP
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

# Impelementation of the final cross attention layer
class PureAttentionPoolingBlock(nn.Module):
    """
    Just a pure attn_pooling implementation, without ln_post, without projection, no mormalized_final
    """

    def __init__(
            self,
            context_dim: int,
            n_head: int = 8,
            norm_layer = LayerNorm,
            need_weights: bool = False
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True,
                                          add_zero_attn=True)
        #self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(context_dim)
        self.ln_k = norm_layer(context_dim)
        self.ln_v = norm_layer(context_dim)
        self.need_weights=need_weights
        self.n_head = n_head

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, output_attn_weights=False, average_attn_weights=True, mask = None):
        batch_size, seg_length, embed_dim = k.size()
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        themask = None if mask is None else (1-mask).bool()

        if self.need_weights or output_attn_weights:
            out, attn_weights = self.attn(q, k, v, need_weights=True, average_attn_weights=average_attn_weights, attn_mask = themask)
            return out, attn_weights
        else:
            out = self.attn(q, k, v, need_weights=False,attn_mask = themask)[0]
        # we can directly normalize the output, without setting a flag
        #return F.normalize(out, dim=-1)
            return out

# ItemizedCLIP model
class ItemizedCLIP(CustomTextCLIP):
    def __init__(self,clip_dim = 512,**kwargs):
        super().__init__(**kwargs)
        self.visual_attn = PureAttentionPoolingBlock(clip_dim)

    def encode_image(self, image, normalize: bool = False):
        all_tokens = self.visual(image)
        global_image_token, local_image_tokens = all_tokens[:,0],all_tokens[:,1:]
        return global_image_token, local_image_tokens
        
    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features,image_tokens= self.encode_image(image,normalize=False) if image is not None else (None,None)
        text_features = self.encode_text(text,normalize=False) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features, # global image feature
                "text_features": text_features, # text features for each text item
                "image_tokens": image_tokens, # local image features
                "logit_scale": self.logit_scale.exp(), # the logit scale
                "visual_proj": self.visual_attn # the crossattn layer
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, (global_text_features,local_text_features), self.logit_scale.exp(), self.logit_bias
        return image_features, (global_text_features,local_text_features), self.logit_scale.exp()
