"""
CT-Rate dataset, adapted from HLIP repository: https://github.com/Zch0414/hlip/blob/master/src/hlip_train/data_chest_ct.py
"""

import os
import sys
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))

from torchvision.transforms import Normalize
from open_clip_train.data import *
from open_mri_train.data import chunked_collator
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
random.seed(42)

CT_RATE_INVALID_DATA = ['train_1267_a_4', 'train_11755_a_3', 'train_11755_a_4']
JSON_ROOT = '../open_ct_rate'
DATA_ROOT = '/data/ct_rate/'

# Information about a single study
class StudyInfo(object):
    def __init__(self, root, key, value):
        self.scans = []
        for scan in value['recons']:
            scan = scan.rsplit('.', 2)[0]
            if scan in CT_RATE_INVALID_DATA:
                continue
            else:
                self.scans.append(os.path.join(root, key.rsplit('_', 1)[0], key, scan + '.pt'))
        self.scans = np.array(self.scans)
        self.report = np.array(value['report'])

    def get_report(self, shuffle):
        if shuffle: 
            return np.random.permutation(self.report).tolist()
        else:
            return self.report.tolist()

    def get_scans(self, shuffle):
        if shuffle: # this is for training
            return np.random.permutation(self.scans).tolist()
        else:
            return self.scans.tolist()

# The CTRate dataset
class StudyDataset(Dataset):
    def __init__(
        self, 
        json_root, data_root, input_filename, input_info,
        transform=None,
        tokenizer=None,
        num_samples = 6,
        randcrop = True
    ):
        with open(os.path.join(json_root, input_filename + '.json'), 'r') as file:
            studies = json.load(file)
        self.studies = [StudyInfo(root=os.path.join(data_root, 'train'), key=key, value=value) for key, value in studies.items()]
        
        self.input_info = (float(input_info[0]), float(input_info[1]), str(input_info[2]))
        self.transform = transform
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.randcrop = randcrop

    def __len__(self):
        return len(self.studies)
    
    def __getitem__(self, idx):
        study = self.studies[idx]

        textmask = torch.ones(self.num_samples)

        # get report
        report = study.get_report(shuffle=self.randcrop)

        is_normal = len(report) == 1 and ('no significant abnormalities' == report[0] or 'no abnormalities detected' == report[0]) 

        #print(report)
        num_items = len(report)
        if num_items > self.num_samples:
            report = report[0:self.num_samples]
        if num_items < self.num_samples:
            tofill = self.num_samples - num_items
            report += ['Placeholder'] * tofill
            textmask[num_items:] = 0
        report = [r.lower() for r in report]

        


        report = self.tokenizer(report)
        

        # get scan
        scan = study.get_scans(shuffle=True)[0] # CT-RATE is a curated dataset
        
        # load in scan
        img = torch.load(scan, weights_only=True)
        img = (img - self.input_info[0]) / (self.input_info[1] - self.input_info[0])
        img = torch.clip(img, 0., 1.)
        img = img[None, ...].float() # [1, d, h, w]

        # transform
        if self.transform:
            img = self.transform(img)
            img = torch.as_tensor(img).float()
        else: 
            if self.input_info[2] == "crop":
                # pad
                _, d, h, w = img.shape
                pad_d = max(112 - d, 0)
                pad_h = max(336 - h, 0)
                pad_w = max(336 - w, 0)
                if self.randcrop:
                    pd = random.randint(0,pad_d)
                    ph = random.randint(0,pad_h)
                    pw = random.randint(0,pad_w)
                else:
                    pd = pad_d // 2
                    ph = pad_h // 2
                    pw = pad_w // 2
                pad_d1, pad_d2 = pd, pad_d - pd
                pad_h1, pad_h2 = ph, pad_h - ph
                pad_w1, pad_w2 = pw, pad_w - pw
                img = torch.nn.functional.pad(
                    img[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2),
                    mode='constant', 
                    value=0
                ).squeeze(0)
                
                # crop [hard code]: tuning this is not interesting
                _, d, h, w = img.shape
                if self.randcrop:
                    start_d = random.randint(0,d-112)
                    start_h = random.randint(0,h-336)
                    start_w = random.randint(0,w-336)
                else:
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

        return img[None, ...], report, is_normal, torch.zeros(1), textmask 

# Get Training dataset
def get_train_dataset(args, preprocess_fn, tokenizer=None):
    input_filename = args.train_data
    assert input_filename
    dataset = StudyDataset(
        JSON_ROOT, DATA_ROOT, input_filename,
        args.input_info,
        preprocess_fn,
        tokenizer,
        args.num_sampled_captions
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        collate_fn = chunked_collator
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

# Get zero-shot evaluation dataset
def get_zeroshot_ct_rate_dataset(args, preprocess_fn):
    from .zeroshot_ct_rate import get_data
    dataloader = get_data(args, preprocess_fn)
    dataloader.num_samples = len(dataloader.dataset)
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, None)

# Get training and zero-shot datasets
def get_ctrate_data(args, tokenizer=None):
    data = {}
    if args.train_data:
        data["train"] = get_train_dataset(args, None, tokenizer=tokenizer)
    if args.zeroshot_ct_rate:
        data["zeroshot-ct-rate"] = get_zeroshot_ct_rate_dataset(args, None)
    return data

