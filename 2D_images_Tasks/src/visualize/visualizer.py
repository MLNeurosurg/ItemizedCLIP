import torch.nn.functional as F
import sys
import json

# Generate attention-score based visualizations for ItemizedCLIP
def visualize_itemizedclip(model, data, args, autocast, limit=100):
    vloader = data['val'].dataloader
    if hasattr(vloader, 'dataset'):
        vloader.dataset.retraw = True
    ec = json.load(open(args.external_captions if ';' not in args.external_captions else args.external_captions.split(';')[-1]))
    external_captions = {int(k):ec[k] for k in ec}
    device = 'cuda'
    for idx,batch in enumerate(vloader):
        if args.viz_id is not None and idx != args.viz_id:
            if idx > args.viz_id:
                break
            else:
                continue
        if idx >= limit and args.viz_id is None:
            break
        images, texts, textmask, textidx = batch
        k = textidx
        if 'rsicd' in args.val_data:
            textidx,_ = textidx
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        textmask = textmask.to(device=device, non_blocking=True)
        with autocast():
            model_out = model(images, texts)
            lens = textmask.sum().long()
            txtfeats = model_out['text_features']
            txtfeats = txtfeats[:lens]
            sims, attnweights = visualize_image_texts(model,model_out["image_tokens"][0],txtfeats)
        plt_visualize(k,sims,attnweights,external_captions,is_rsicd = 'rsicd' in args.val_data)

def visualize_image_texts(model,imgfeat,txtfeats):
    txtfeats = F.normalize(txtfeats,dim=-1)
    img_features_after_conditioning, attnweights = model.visual_proj(
                    txtfeats.unsqueeze(0),
                    imgfeat.unsqueeze(0),
                    imgfeat.unsqueeze(0),
                    output_attn_weights=True
                )
    img_features_after_conditioning = F.normalize(img_features_after_conditioning, dim=-1)
    sims = (img_features_after_conditioning[0] @ txtfeats.t())[0]
    return sims,attnweights

def plt_visualize(keys,sims,attnweights,external_captions,is_rsicd = False):
    
    if is_rsicd:
        idx,imgpath = keys
        idx = idx.item()
    else:
        idx,key = keys[0]
        idx = idx // 10000 / 2 * 10000 + idx % 10000
        imgpath = '/nfs/turbo/umms-tocho/code/yiwei/flair/tempview/'+key+'.jpg'
    origtexts = external_captions[idx]
    show_attention_visualizations(imgpath,origtexts,sims,attnweights[:,:,:-1].view(-1,14,14).detach().cpu(),str(idx)+'-'+key if not is_rsicd else str(idx))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch,json,textwrap

def upsample_attention_nearest(attn_map, patch_size=16):
    """
    Upsamples a 14x14 attention map to 224x224 using nearest-neighbor block tiling.
    """
    return np.kron(attn_map, np.ones((patch_size, patch_size)))

# save attention visualizations to specified location
def show_attention_visualizations(image_path, origtexts, sims, attention_maps,savename, cmap='Reds', alpha=0.5, savedir = 'visualize/visualizations'):
    """
    Parameters:
        image_path: str, path to the image (jpg)
        attention_maps: List or array of N (14x14) attention maps
        cmap: matplotlib colormap (e.g., 'Reds')
        alpha: transparency for heatmap overlay
    """
    plt.clf()
    # Load and resize image to 224x224
    if isinstance(image_path,str):
        img = Image.open(image_path)
        img = img.convert("RGB").resize((224, 224))
        img_np = np.array(img)
    else:
        img_np = image_path.numpy()[0].transpose((1,2,0))
    
    attention_maps /= attention_maps.sum()

    zero_attention_maps = torch.zeros_like(attention_maps[0:1])
    attention_maps = torch.cat([attention_maps,zero_attention_maps],dim=0)
    

    N = len(attention_maps)
    ncols = min(N, 3)
    nrows = (N + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)


    for idx, attn in enumerate(attention_maps):
        ax = axes[idx]
        ax.imshow(img_np)

        # Grid-based upsampling: each 1x1 becomes 16x16 block
        attn_resized = upsample_attention_nearest(attn)
        ax.imshow(attn_resized, cmap=cmap, alpha=alpha if idx < len(origtexts) else 0, extent=(0, 224, 224, 0))
        if idx < len(origtexts):
            wrapped = "\n".join(textwrap.wrap(origtexts[idx], 25))
            ax.set_title(wrapped+'\n Similarity: '+str(sims[idx].item()))
        ax.axis('off')

    for ax in axes[N:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(savedir+'/'+savename+'.jpg')

