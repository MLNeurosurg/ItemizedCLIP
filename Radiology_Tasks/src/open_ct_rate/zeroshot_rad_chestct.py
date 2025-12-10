"""
Zero-shot evaluation on Rad-ChestCT dataset
Adapted from HLIP Repository: https://github.com/Zch0414/hlip/blob/master/src/hlip_test/zeroshot_rad_chestct.py
Included modifications for allowing TCSim-based zero-shot
"""

import os
import sys
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))

from pathos.multiprocessing import ProcessingPool as Pool
import math
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from open_mri.factory import create_model_and_transforms, get_tokenizer
from open_clip import  get_input_dtype, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from open_ct_rate import visual_encoder
from open_ct_rate.zeroshot_metadata_rad_chestct import CLASSNAMES, ORGANS

import torch.nn.functional as F

def get_args_parser():
    parser = argparse.ArgumentParser('Perform Zero-shot', add_help=False)
    parser.add_argument('--model', default='CTRATE_HLIP_BiomedBERT_itemizedclip', type=str)
    parser.add_argument('--use-cxr-bert', default=True, action='store_false')
    parser.add_argument('--lora-text', default=False, action='store_true')
    parser.add_argument('--resume', default='/checkpoints/epoch_68.pt', type=str)

    parser.add_argument('--data-root', default='/nfs')
    parser.add_argument('--zeroshot-rad-chestct', default='../open_ct_rate/rad_chestct_labels.csv', type=str)
    parser.add_argument('--input-info', nargs='+', default=["-1150", "350", "crop"])
    parser.add_argument('--zeroshot-template', default='volume', type=str)

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--workers', default=8, type=int)

    return parser


def random_seed(seed=0, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_data(args, preprocess_fn=None):
    class ZeroShotDataset(Dataset):
        def __init__(
            self,
            root, input_filename, input_info,
            transform=None,
        ):
            self.cts = []
            df = pd.read_csv(input_filename)
            for _, row in df.iterrows():
                recon = row['NoteAcc_DEID']
                self.cts.append((os.path.join(root, recon + '.pt'), row[CLASSNAMES].astype(int).tolist()))
            
            self.input_info = (float(input_info[0]), float(input_info[1]), str(input_info[2]))
            self.transform = transform

        def __len__(self):
            return len(self.cts)

        def __getitem__(self, idx):
            recon, target = self.cts[idx]

            img = torch.load(recon, weights_only=True)
            img = (img - self.input_info[0]) / (self.input_info[1] - self.input_info[0])
            img = torch.clip(img, 0., 1.)
            img = img[None, ...].float()

            if self.transform:
                img = self.transform(img)
                img = torch.as_tensor(img).float()
            else: 
                if self.input_info[2] == "crop":
                    _, d, h, w = img.shape
                    pad_d = max(112 - d, 0)
                    pad_h = max(336 - h, 0)
                    pad_w = max(336 - w, 0)
                    pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2
                    pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
                    pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
                    img = torch.nn.functional.pad(
                        img[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2),
                        mode='constant', 
                        value=0
                    ).squeeze(0)
                    
                    _, d, h, w = img.shape
                    start_d = (d - 112) // 2
                    start_h = (h - 336) // 2
                    start_w = (w - 336) // 2
                    img = img[
                        :, 
                        start_d:start_d + 112,
                        start_h:start_h + 336,
                        start_w:start_w + 336
                    ]

                elif self.input_info[2] == "resize":
                    img = torch.nn.functional.interpolate(img[None, ...], size=(112, 336, 336), mode='trilinear').squeeze(0)
                
                else:
                    raise NotImplementedError

            # normalize
            normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
            img = normalizer(img)

            return recon, img[None, ...], torch.as_tensor(target)
    

    dataset = ZeroShotDataset(
        args.data_root, args.zeroshot_rad_chestct, args.input_info,
        preprocess_fn
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    return dataloader


def findthresholdhelp(inp):
    return find_threshold(inp[0],inp[1])

def find_threshold(y_true, y_score):
    """
    Copy from https://github.com/alibaba-damo-academy/fvlm/blob/d768ec1546fb825fcc9ea9b3e7b2754a69f870c1/calc_metrics.py#L8C1-L8C32
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        y_true (numpy.ndarray): True labels.
        y_score (numpy.ndarray): Predicted probabilities.

    Returns:
        float: Optimal threshold.
    """

    best_threshold = 0
    best_roc = 10000

    # Iterate over potential thresholds
    thresholds = np.linspace(-1, 1, 4000)
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype(int)
        confusion = confusion_matrix(y_true, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        curr_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if curr_roc <= best_roc:
            best_roc = curr_roc
            best_threshold = threshold

    return best_threshold


def get_text(disease,organ):
    if disease.lower() == 'lung nodule':
        disease = 'nodules or nodular density'
    if disease.lower() == 'lung opacity':
        disease = 'opacity'
    if disease.lower() == 'bronchiectasis':
        disease = 'bronchiectasis or bronchiectatic changes'
    return ["findings consistent with "+disease.lower()]


def get_text_embed(text,tokenizer,model, args):
    report = tokenizer(text).to(args.device)
    modelout = model(None,report)["text_features"]
    return F.normalize(modelout,dim=-1)

def zero_shot(model, tokenizer, dataloader, args):
    
    model.eval()
    device = torch.device(args.device)
    autocast = get_autocast('amp', device_type=device.type)
    input_dtype = get_input_dtype('amp')
    # build classifier
    with autocast():
        texts = [get_text(classname,ORGANS[classname]) for classname in CLASSNAMES]
        numtextperclass = len(texts[0])
        alltexts = []
        for i in range(numtextperclass):
            alltexts += [t[i] for t in texts]
        textembs = get_text_embed(alltexts,tokenizer,model,args)

    with torch.inference_mode():
        labels = {key: [] for key in CLASSNAMES}
        logits = {key: [] for key in CLASSNAMES}
        
        for batch in tqdm(dataloader, total=len(dataloader)):
            recon, image, target = batch
            image = image.to(device=device, dtype=input_dtype)
            target = target.to(device)
            
            for idx in range(target.shape[1]):
                labels[CLASSNAMES[idx]].append(target[0, idx].cpu().float().item())
            
            with autocast():
                output = model(image,None)
                image_features = output['image_tokens'] # BxSx512
                
                image_features = F.normalize(image_features,dim=-1)
                attn_out = output['visual_proj'](textembs.unsqueeze(0),image_features,image_features) # BxCx512
                attn_out = F.normalize(attn_out,dim=-1)
                
                sims = torch.einsum('bck,ck -> bc',attn_out,textembs)
                
                # predict
                for i,key in enumerate(CLASSNAMES):
                    logits[key].append([sims[0,i+j*len(CLASSNAMES)].cpu().float().item() for j in range(numtextperclass)])



        results = {key: {} for key in CLASSNAMES}
        results['* mean'] = {}
        inppairs = []
        for j in range(numtextperclass):
            for key in CLASSNAMES:
                inppairs.append((np.array(labels[key]),np.array(logits[key])[:,j]))

        with Pool(8) as p:
            threshes = list(tqdm(p.imap(findthresholdhelp,inppairs)))

        counter = 0
        for j in range(numtextperclass):
            mean_auc, mean_acc, mean_weighted_f1, mean_recall, mean_precision = 0., 0., 0., 0., 0.

            for key in CLASSNAMES:
                threshold = threshes[counter]
                counter += 1

                auc = roc_auc_score(np.array(labels[key]), np.array(logits[key])[:,j]) 
                mean_auc += auc / len(CLASSNAMES)

                acc = accuracy_score(np.array(labels[key]), (np.array(logits[key])[:,j] > threshold).astype(int)) 
                mean_acc += acc / len(CLASSNAMES)

                weighted_f1 = f1_score(np.array(labels[key]), (np.array(logits[key])[:,j] > threshold).astype(int), average='weighted')
                mean_weighted_f1 += weighted_f1 / len(CLASSNAMES)

                recall = recall_score(np.array(labels[key]), (np.array(logits[key])[:,j] > threshold).astype(int)) 
                mean_recall += recall / len(CLASSNAMES)

                precision = precision_score(np.array(labels[key]), (np.array(logits[key])[:,j] > threshold).astype(int)) 
                mean_precision += precision / len(CLASSNAMES)

                results[key].update({
                    'auc '+str(j): auc,
                    '* acc (balanced)'+str(j): acc,
                    '* f1 (weighted)'+str(j): weighted_f1,
                    '* recall'+str(j): recall,
                    '* precision'+str(j): precision,
                })
            results['* mean'].update({
                'zero shot mean auc'+str(j): mean_auc,
                'zero shot * mean acc (balanced)'+str(j): mean_acc,
                'zero shot * mean f1 (weighted)'+str(j): mean_weighted_f1,
                'zero shot * mean recall'+str(j): mean_recall,
                'zero shot * mean precision'+str(j): mean_precision,
            })

    return results


def main(args):


    random_seed(0, 0)

    # create model
    for _c in os.listdir('../open_mri/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('../open_mri/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    model, _, _ = create_model_and_transforms(args.model, device=args.device, precision='amp', output_dict=True, my_custom_model=True)
    
    # replace with cxr_bert
    if args.use_cxr_bert:
        from transformers import AutoModel
        cxr_bert = AutoModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', trust_remote_code=True).bert
        cxr_bert.to(device=args.device)
        model.text.transformer = cxr_bert
    
    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)

    # create dataset
    data = get_data(args, None)

    # zero shot
    results = zero_shot(model, tokenizer, data, args)
    print(results['* mean'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform Zero-shot', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
