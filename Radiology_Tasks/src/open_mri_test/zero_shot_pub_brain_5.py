"""
Adapted from HLIP repository: https://github.com/Zch0414/hlip/blob/master/src/hlip_test/zeroshot_pub_brain_5.py
"""

import os
import sys
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
from open_mri.factory import create_model_and_transforms,  get_tokenizer
from open_clip import get_input_dtype, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from open_mri_train.data import chartovec

def get_args_parser():
    parser = argparse.ArgumentParser('Perform Zero-shot', add_help=False)
    # Model
    parser.add_argument('--model', default='HLIPSN_BiomedBERT_81616_itemizedclip', type=str)
    parser.add_argument('--resume', default='checkpoints/epoch_13.pt', type=str)
    parser.add_argument('--precision', default='amp', type=str)
    parser.add_argument('--num-series', default=10, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--data-dir', default='/nfs', type=str)

    # Data
    parser.add_argument('--num-slices', default=48, type=int)
    parser.add_argument('--input-filename',default = 'pub_brain_5.csv', type=str)

    #task
    
    parser.add_argument('--tasks', default='stroke,glioma,meningioma,metastasis,tumor,disease', type=str)

    # Device
    parser.add_argument('--device', default='cuda', type=str)


    return parser

# Get the pub_brain_5 dataset based on the csv
def get_data(args, preprocess_fn):
    class PubBrain5Dataset(Dataset):
        def __init__(
            self,
            input_filename,
            num_slices, transform=None,
        ):
            self.studies = []
            df = pd.read_csv(input_filename)
            self.labelmaps = {}
            for _, row in df.iterrows():
                if len(os.listdir(args.data_dir+row['study'])):
                    self.studies.append(row['study'])
                    self.labelmaps[row['study']] = torch.LongTensor([row['Normal'],row['Stroke'],row['Glioma'],row['Meningioma'],row['Metastasis']])
                else:
                    print(row['study'])


            self.num_slices = num_slices
            self.transform = transform
        
        def __len__(self):
            return len(self.studies)

        def __getitem__(self, idx):
            study = self.studies[idx]

            # load in imgs
            imgs = []
            serienames = []
            for scan in [os.path.join('/nfs/turbo/umms-tocho-snr/exp/chuizhao'+study, p) for p in os.listdir('/nfs/turbo/umms-tocho-snr/exp/chuizhao'+study)]:
                img = torch.load(scan, weights_only=True).float()/255

                lastp = scan.split('/')[-1][:-3]
                lastp = lastp.split('-')[-1]
                serienames.append('AX_'+lastp)

                # check
                if len(img.shape) == 4:
                    img = img[:, :, :, 0]

                img = img[None, ...].float()

                # process
                if self.transform:
                    img = self.transform(img)
                    img = torch.as_tensor(img).float()
                else:
                    _, _, h, w = img.shape

                    # padding to the longest side.                
                    size = max(h, w)
                    pad_h = size - h; pad_w = size - w
                    left = pad_w // 2; right = pad_w - left; top = pad_h // 2; bottom = pad_h - top
                    img = torch.nn.functional.pad(img, (left, right, top, bottom), mode="constant", value=0)

                    # resize to 256, crop to 224
                    img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bilinear')
                    img = torch.nn.functional.interpolate(img[None, ...], size=(self.num_slices, 256, 256), mode='nearest-exact').squeeze(0)
                    img = img[:, :, 16:240, 16:240]

                # normalize
                normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
                img = normalizer(img)
                imgs.append(img) 

            #print(serienames)
            flist = [chartovec(s) for s in serienames]
            serienames = torch.stack(flist,dim=0)
            return study, torch.stack(imgs, dim=0), serienames, self.labelmaps[study]
    
    dataset = PubBrain5Dataset(args.input_filename, args.num_slices, preprocess_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader

def get_text_embed(text,tokenizer,model, args):
    report = tokenizer([text]).to(args.device)
    modelout = model(None,report)["text_features"][0]
    assert len(modelout) == 512
    return F.normalize(modelout,dim=-1)


def get_visual_embed(model,image,sn, args):
    device = torch.device(args.device)
    image = image.to(device=device, non_blocking=True)
    sn = sn.to(device=device, non_blocking=True)
    outs = model((image,sn),None)
    return F.normalize(outs['image_tokens'][0],dim=-1),outs['visual_proj']

def get_logits(model,tokenizer,dataloader,autocast, args):
    texts = ["Study is unremarkable", "Acute stroke", "Glioma", "Meningioma", "Metastasis"]
    with autocast():
        textembs = torch.stack([get_text_embed(text,tokenizer,model,args) for text in texts])

    logitslist = []
    for _,image,sn,labels in tqdm(dataloader):
        with autocast():
            visembed,visproj = get_visual_embed(model,image,sn, args)
            outemb = visproj(textembs.unsqueeze(0),visembed.unsqueeze(0),visembed.unsqueeze(0))[0]
            outemb = F.normalize(outemb,dim=-1)
        logits = torch.einsum('bk,bk->b',textembs,outemb) # 5
        logitslist.append((logits,labels.squeeze().detach().cpu().numpy()))
    return logitslist

def zero_shot(logitslist,args):

    # stroke detection (2-way)
    if 'stroke' in args.tasks:
        with torch.inference_mode():
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for blogits,blabels in tqdm(logitslist):

                if blabels[0] == 0 and blabels[1] == 0:
                    continue
                ground_truth.append(blabels[1])
                predlogits = blogits[[0,1]].softmax(dim=-1)
                logits.append(predlogits[1].detach().cpu().numpy())
                preds.append(predlogits.argmax(-1).item())

            # metrics
            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)

            acc = balanced_accuracy_score(ground_truth, preds)
            f1 = f1_score(ground_truth, preds, average='weighted')
            recall = recall_score(ground_truth, preds)
            precision = precision_score(ground_truth, preds)
            auc = roc_auc_score(ground_truth, logits)
            print(f"Stroke Detection (2-way): Balanced Acc: {acc}; Weghted F1: {f1}; Recall: {recall}; Precision: {precision}; Macro AUC: {auc}")

    # Glioma detection (2-way)
    if 'glioma' in args.tasks:
        with torch.inference_mode():
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for blogits,blabels in tqdm(logitslist):

                if blabels[0] == 0 and blabels[2] == 0:
                    continue
                ground_truth.append(blabels[2])
                predlogits = blogits[[0,2]].softmax(dim=-1)
                logits.append(predlogits[1].detach().cpu().numpy())
                preds.append(predlogits.argmax(-1).item())

            # metrics
            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)

            acc = balanced_accuracy_score(ground_truth, preds)
            f1 = f1_score(ground_truth, preds, average='weighted')
            recall = recall_score(ground_truth, preds)
            precision = precision_score(ground_truth, preds)
            auc = roc_auc_score(ground_truth, logits)
            print(f"Glioma Detection (2-way): Balanced Acc: {acc}; Weighted F1: {f1}; Recall: {recall}; Precision: {precision}; Macro AUC: {auc}")

    # Meningioma detection (2-way)
    if 'meningioma' in args.tasks:
        with torch.inference_mode():
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for blogits,blabels in tqdm(logitslist):

                if blabels[0] == 0 and blabels[3] == 0:
                    continue
                ground_truth.append(blabels[3])
                predlogits = blogits[[0,3]].softmax(dim=-1)
                logits.append(predlogits[1].detach().cpu().numpy())
                preds.append(predlogits.argmax(-1).item())

            # metrics
            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)

            acc = balanced_accuracy_score(ground_truth, preds)
            f1 = f1_score(ground_truth, preds, average='weighted')
            recall = recall_score(ground_truth, preds)
            precision = precision_score(ground_truth, preds)
            auc = roc_auc_score(ground_truth, logits)
            print(f"Meningioma Detection (2-way): Balanced Acc: {acc}; Weighted F1: {f1}; Recall: {recall}; Precision: {precision}; Macro AUC: {auc}")

    # Metastasis detection (2-way)
    if 'metastasis' in args.tasks:
        with torch.inference_mode():
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for blogits,blabels in tqdm(logitslist):

                if blabels[0] == 0 and blabels[4] == 0:
                    continue
                ground_truth.append(blabels[4])
                predlogits = blogits[[0,4]].softmax(dim=-1)
                logits.append(predlogits[1].detach().cpu().numpy())
                preds.append(predlogits.argmax(-1).item())

            # metrics
            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)

            acc = balanced_accuracy_score(ground_truth, preds)
            f1 = f1_score(ground_truth, preds, average='weighted')
            recall = recall_score(ground_truth, preds)
            precision = precision_score(ground_truth, preds)
            auc = roc_auc_score(ground_truth, logits)
            print(f"Metastasis Detection (2-way): Balanced Acc: {acc}; Weighted F1: {f1}; Recall: {recall}; Precision: {precision}; Macro AUC: {auc}")

    # brain tumor classification (3-way)
    if 'tumor' in args.tasks:
        with torch.inference_mode():
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for blogits,blabels in tqdm(logitslist):

                if blabels[2] == 0 and blabels[3] == 0 and blabels[4] == 0:
                    continue
                ground_truth.append(blabels[2:5].argmax(-1).item())
                predlogits = blogits[2:5].softmax(dim=-1)
                logits.append(predlogits.detach().cpu().numpy())
                preds.append(predlogits.argmax(-1).item())

            # metrics
            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)

            acc = balanced_accuracy_score(ground_truth, preds)
            print(f"Tumor Classification (3-way): Balanced Acc: {acc}")

    # brain disease classification (5-way)
    if 'disease' in args.tasks:
        with torch.inference_mode():
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for blogits,blabels in tqdm(logitslist):

                ground_truth.append(blabels.argmax(-1).item())
                predlogits = blogits.softmax(dim=-1)
                logits.append(predlogits.detach().cpu().numpy())
                preds.append(predlogits.argmax(-1).item())

            # metrics
            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)
            acc = balanced_accuracy_score(ground_truth, preds)
            print(f"Disease Classification (5-way): Balanced Acc: {acc}")
    return

def main(args):
    print(f'Perform Zero-shot for {args.tasks}:\n')
    
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
    tokenizer = get_tokenizer(args.model)
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)


    # Zero shot
    with torch.inference_mode():
        dataloader = get_data(args,None)
        logitslist = get_logits(model,tokenizer,dataloader,autocast,args)
        zero_shot(logitslist, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform Zero-shot', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
