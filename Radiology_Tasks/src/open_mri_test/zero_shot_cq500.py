"""
Code for zero-shot evaluation of CQ500
"""

import os
import sys,csv
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
import json
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from open_mri import hlip
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, balanced_accuracy_score
import numpy as np
from open_clip import  get_input_dtype, build_zero_shot_classifier
from open_mri.factory import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast
from open_mri_test.data import get_public_dataset

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from open_mri_train.data import chartovec

def get_args_parser():
    parser = argparse.ArgumentParser('Perform Zero-shot', add_help=False)
    # Model
    parser.add_argument('--dataset',default = 'cq500', type=str)
    parser.add_argument('--data-dir', default='/nfs', type=str)
    parser.add_argument('--model', default='HLIPSN_BiomedBERT_81616_itemizedclip', type=str)
    parser.add_argument('--resume', default='/checkpoints/epoch_10.pt', type=str)
    parser.add_argument('--precision', default='amp', type=str)
    parser.add_argument('--num-series', default=10, type=int)
    parser.add_argument('--use-serienames', default=False, action='store_true')
    parser.add_argument('--bloodonly', default=False, action='store_true') # only use bloodview
    parser.add_argument('--workers', default=8, type=int)

    # Data
    parser.add_argument('--num-slices', default=48, type=int)


    # Device
    parser.add_argument('--device', default='cuda', type=str)


    return parser

# Get Text Embeddings
def get_text_embed(text,tokenizer,model, args):
    report = tokenizer([text]).to(args.device)
    modelout = model(None,report)["text_features"][0]
    assert len(modelout) == 512
    return F.normalize(modelout,dim=-1)

# Get Visual Embeddings
def get_visual_embed(model,image,sn, args):
    device = torch.device(args.device)
    image = image.to(device=device, non_blocking=True)
    sn = sn.to(device=device, non_blocking=True)
    outs = model((image,sn),None)
    return F.normalize(outs['image_tokens'][0],dim=-1),outs['visual_proj']

# Get zero-shot logits
def get_logits(model,tokenizer,autocast, args, testlist, dataloader):
    texts = ["Intracranial hemorrhage", "Intracranial hemorrhage with intraparenchymal hemorrhage", "Intracranial hemorrhage with intraventricular hemorrhage", "Intracranial hemorrhage with subdural hemorrhage", "Intracranial hemorrhage with epidural hemorrhage", "Intracranial hemorrhage with subarachnoid hemorrhage", "Bleeding in left side", "Bleeding in right side","Midline shift", "Mass Effect"]
    with autocast():
        textembs = torch.stack([get_text_embed(text,tokenizer,model,args) for text in texts])
    
    logitslist = torch.zeros(len(testlist),10)

    for uid,image,_,_,sn in tqdm(dataloader):
        uid = uid[0]
        if uid in testlist:
            with torch.no_grad():
                with autocast():
                    visembed,visproj = get_visual_embed(model,image,sn, args)
                    outemb = visproj(textembs.unsqueeze(0),visembed.unsqueeze(0),visembed.unsqueeze(0))[0]
                    outemb = F.normalize(outemb,dim=-1)
            logits = torch.einsum('bk,bk->b',textembs,outemb) # 10
            logitslist[testlist.index(uid)] = logits

    return logitslist

# Get zero-shot AUC
def zero_shot(logitslist,labels,args):

    for i in range(10):
        l = logitslist[:,i].detach().cpu().numpy()
        la = labels[:,i]
        print(roc_auc_score(la,l))

def main(args):
    
    # Create model
    for _c in os.listdir('../open_mri/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('../open_mri/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    model, _, _ = create_model_and_transforms(args.model, device=args.device, precision=args.precision, output_dict=True, my_custom_model = True)
    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    model = model.cuda()
    tokenizer = get_tokenizer(args.model)
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)

    reader = csv.reader(open('CQ500/CQ500.csv'))
    labeldict = {}

    labelids = [11,15,19,23,27,31,35,39,59,63]

    for i,row in enumerate(reader):
        if i == 0:
            print([row[idx] for idx in labelids])
        else:
            labeldict[row[0]] = [int(row[idx]) for idx in labelids]


    dataloader = get_public_dataset(args, None, tokenizer)
    testlist = list(dataloader.dataset.mykeys)
    labels = np.asarray([labeldict[key] for key in testlist])

    # Zero shot
    with torch.inference_mode():
        logitslist = get_logits(model,tokenizer,autocast,args, testlist,dataloader)
        zero_shot(logitslist, labels, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform Zero-shot', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
