"""
This code is adapted from Prima GitHub repository: https://github.com/MLNeurosurg/Prima/blob/main/Prima_training_and_evaluation/model_parts.py
The code contains implementation of the character-level transformer encoder for series names, used for HLIP-SN
"""

import torch
import math
from torch import nn

from positional_encodings.torch_encodings import PositionalEncoding1D
# helpers
from einops import rearrange, repeat

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, culen = None,mxlen = None):
        if culen is None:
            return self.fn(self.norm(x))
        return self.fn(self.norm(x), culen,mxlen)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
except:
    print('warning: flash attn not loaded')
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., causal = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.inner_dim = inner_dim
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.dropoutp = dropout
        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x, culen, mxlen):
        if not hasattr(self,'causal'):
            self.causal = False
        if culen is None:
            b,s,_ = x.size()
            qkv = self.to_qkv(x).view(b,s,3,self.heads,self.dim_head)
            if self.hasattr('noflashattn') and self.noflashattn:
                out = no_flash_attn_substitute(qkv)
            else:
                out = flash_attn_qkvpacked_func(qkv.half(),dropout_p = self.dropoutp).float()

            out = out.flatten(start_dim=2)
            return self.to_out(out)
        bxs,embsize = x.size()
        qkv = self.to_qkv(x).view(bxs,3,self.heads,self.dim_head)
        if hasattr(self,'noflashattn') and self.noflashattn:
            out = no_flash_attn_varlen_substitute(qkv,culen.type(torch.int32))
        else:
            out = flash_attn_varlen_qkvpacked_func(qkv,culen.type(torch.int32),mxlen,dropout_p = self.dropoutp, causal=self.causal) # flash attention!
        out = out.flatten(start_dim=1)
        try:
            assert len(out.size()) == 2
            assert out.size()[-1] == self.inner_dim
        except:
            print(out.size())
            exit(1)
        return self.to_out(out)

def no_flash_attn_substitute(qkv):
    qkv = qkv.transpose(0,2).transpose(1,2)
    q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d'), qkv)

    d = qkv.size()[-1]
    dots = torch.matmul(q, k.transpose(-1, -2)) * (d ** -0.5)

    attn = torch.nn.functional.softmax(dots)

    out = torch.matmul(attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return out

def no_flash_attn_varlen_substitute(qkv,culen):
    qkv = qkv.transpose(0,1)
    q, k, v = map(lambda t: rearrange(t, 'n h d -> h n d'), qkv)
    
    n = qkv.size()[1]
    h = qkv.size()[2]
    d = qkv.size()[-1]
    out = torch.zeros(h,n,d).to(qkv.device)
    for i in range(len(culen)-1):
        dots = torch.matmul(q[:,culen[i]:culen[i+1]], k[:,culen[i]:culen[i+1]].transpose(-1, -2)) * (d ** -0.5)
        attn = torch.nn.functional.softmax(dots,dim=-1)
        #print(attn)
        out[:,culen[i]:culen[i+1]] = torch.matmul(attn, v[:,culen[i]:culen[i+1]])
    out = rearrange(out, 'h n d -> n (h d)')
    return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., causal = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, causal = causal)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, culen, mxlen):
        for attn, ff in self.layers:
            x = attn(x,culen,mxlen) + x
            x = ff(x) + x
        return x


# the transformer that encodes the serienames
class SerieTransformerEncoder(nn.Module):
    def __init__(self,out_dim,positional_encoding_dim=10):
        super().__init__()
        self.embsize = 256
        self.embed = nn.Embedding(47,246)
        self.transformer = Transformer(dim = 256, depth = 3, heads = 4, dim_head=64, mlp_dim = 300)
        self.linear = nn.Linear(256,out_dim)
        p_enc = PositionalEncoding1D(positional_encoding_dim)
        self.p_enc = p_enc(torch.zeros(1,200,positional_encoding_dim))[0]
        self.prelinear = False
    def forward(self,x):
        occur46 = (x == 46).nonzero()
        assert occur46.size()[0] == len(x)
        assert occur46.size()[1] == 2
        assert torch.all(occur46[:,0] == torch.arange(end=len(x)).to(occur46.device))
        lens = occur46[:,1] + 1

        posenc = self.p_enc[0:x.size()[1]].to(x.device)

        cl = torch.cumsum(lens, dim=0) # cumulative sums of lengths
        culen = torch.zeros(len(cl)+1).long().to(cl.device)
        culen[1:] = cl # cumulative sequence length used for flash attention input
        mxlen = lens.max()
        nx = torch.zeros(culen[-1].item()).long().to(x.device)
        npe = torch.zeros(culen[-1].item(),posenc.size()[-1]).to(x.device)
        for i in range(len(cl)): # move each input sequence into the concatenated long sequence
            assert lens[i] == culen[i+1] - culen[i]
            nx[culen[i]:culen[i+1]] = x[i][0:lens[i]]
            npe[culen[i]:culen[i+1]] = posenc[0:lens[i]]

        nx = self.embed(nx)
        nx = torch.cat([nx,npe],dim=1)
        nx = self.transformer(nx,culen,mxlen)
        xout = torch.stack([nx[culen[i+1]-1] for i in range(len(lens))]) # obtain the last embedding for each text sequence
        if hasattr(self,'prelinear') and self.prelinear:
            return xout
        return self.linear(xout)

    # This function is used to ban flash attention from this module
    def make_no_flashattn(self):
        for layer in self.transformer.layers:
            layer[0].fn.noflashattn = True
