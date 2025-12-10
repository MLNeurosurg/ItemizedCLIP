"""
Adapted from FLAIR: https://github.com/ExplainableML/flair/blob/main/src/flair/data.py

Adapted version loads Itemized-cc0.3M dataset and RSICD dataset, as well as Flickr and MSCOCO retrieval datasets
"""


import json
import logging
import math
import os
import re
import sys
import braceexpand
from dataclasses import dataclass
import random
import torch
import copy
import torchvision
import webdataset as wds
from PIL import Image
from open_clip_train.data import get_dataset_size, detshuffle2, ResampledShards2, tarfile_to_samples_nothrow, SharedEpoch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import collections,io
import pandas as pd
import numpy as np

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def split_caption(text):
    """Split captions by sentence-ending markers."""
    return [cap.strip() for cap in re.split(r'\n|</s>|[.]', text) if cap.strip()]

def random_sample_from_list(captions_list, k, merged_num=1):
    n = len(captions_list)
    if merged_num == 1:
        if n >= k:
            return random.sample(captions_list, k)
        else:  #minimizing caption dupilications
            return random.choices(captions_list, k=k)
            #return captions_list + random.sample(captions_list, k - n)
    elif merged_num >= n:
        return ['. '.join(captions_list)]
    else:
        sampled_list = []
        sampled_indices = draw_numbers(n=n - merged_num, k=k)
        for sampled_index in sampled_indices:
            sampled_list.append('. '.join(captions_list[sampled_index:sampled_index + merged_num]))
        return sampled_list


def draw_numbers(n, k=4):
    population = list(range(0, n))
    if n >= k:
        return random.sample(population, k)
    else:
        return random.choices(population, k=k)


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def draw_numbers(n, k=4):
    population = list(range(0, n))
    if n >= k:
        return random.sample(population, k)
    else:
        return random.choices(population, k=k)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights



def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def filter_no_caption_or_no_image_json(sample):
    has_caption = ('json' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


# This function determines how to sample text items given an image in itemized-cc0.3m. Adapted from FLAIR
def sample_dict(text, k=3, tokenizer=None, sampling_mode='diverse_sampling', pixelprose=False, max_merged_num=3, concat = False,  padto = None, external_captions = None, shuffle = False):
    
    sampled_sentences = []
    if not shuffle and sampling_mode == 'diverse_external':
        sampling_mode = 'external'
    if sampling_mode == 'diverse_sampling' or sampling_mode == 'diverse_external': # with diverse sampling
        if sampling_mode == 'diverse_external':
            if 'index' in text:
                idx = text['index']
            else:
                idx = int(text['key'])
                assert idx % 10000 < 5000
                idx = (idx % 10000) + (idx // 10000)*5000
            if idx in external_captions: # replace cc3m-recap captions with Itemized captions
                sampled_sentences = copy.deepcopy(external_captions[idx])
            else:
                sampled_sentences = text['shortIB_captions']
                if isinstance(sampled_sentences, str):
                    sampled_sentences = [sampled_sentences]
            captions_list = sampled_sentences
            sampled_sentences = []
        else:

            if pixelprose:
                raw_caption = text["caption"]
                captions_list = split_caption(raw_caption)
            else:
                captions_list = (text['raw_caption'] + text['shortIB_captions'] + text['longIB_captions'] +
                                text['shortSV_captions'] + text['longSV_captions'] +
                                text['shortLLA_captions'] + text['longLLA_captions'])
        n_captions = len(captions_list)
        for _ in range(k):
            merged_num = random.randint(1, max_merged_num)
            if merged_num == 1:
                # Sample one caption
                sampled_sentence = random.choice(captions_list)
                sampled_sentences.append(sampled_sentence)
            else:
                prob_flag = 0.5 # 50% merging subsequent captions, 50% merging captions from random positions
                if random.random() < prob_flag:
                    sampled_sentence_list = random_sample_from_list(
                        captions_list, k=1, merged_num=merged_num)
                    sampled_sentences.extend(sampled_sentence_list)
                else:
                    # Randomly select captions to merge
                    if n_captions >= merged_num:
                        captions_to_merge = random.sample(captions_list, merged_num)
                    else:
                        captions_to_merge = [random.choice(captions_list) for _ in range(merged_num)]
                    # Merge the captions
                    sampled_sentence = '. '.join(captions_to_merge)
                    sampled_sentences.append(sampled_sentence)
    elif sampling_mode == 'external': # without diverse sampling
        if 'index' in text:
            idx = text['index']
        else:
            idx = int(text['key'])
            assert idx % 10000 < 5000
            idx = (idx % 10000) + (idx // 10000)*5000
        if idx in external_captions: # replace cc3m-recap captions with Itemized captions
            sampled_sentences = copy.deepcopy(external_captions[idx])
        else:
            sampled_sentences = text['shortIB_captions']
            if isinstance(sampled_sentences, str):
                sampled_sentences = [sampled_sentences]

    else:
        raise NotImplementedError
    

    if shuffle:
        random.shuffle(sampled_sentences)

    
    
    textmask = torch.ones(len(sampled_sentences)).long()
    
    # Pad or truncate the textmask and sampled_sentences to the specified length
    if padto is not None:
        if len(sampled_sentences) < padto:
            textmask = torch.LongTensor([1]*len(sampled_sentences)+[0]*(padto - len(sampled_sentences)))
            sampled_sentences += ["Placeholder"]*(padto - len(sampled_sentences))
        elif len(sampled_sentences) > padto:
            sampled_sentences = sampled_sentences[0:padto]
            textmask = textmask[0:padto]
        
    tokenized_sentences = tokenizer(sampled_sentences)
    return tokenized_sentences, textmask, (int(text['key']),text['key'])


def get_train_val_dataset_fn(dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "rsicd":
        return get_rsicd_dataset
    else:
        raise ValueError(f"Unsupported training dataset type: {dataset_type}")

# The main function for obtaining training and evaluation datasets
def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_train_val_dataset_fn(args.train_dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
    if args.val_data:
        data["val"] = get_train_val_dataset_fn(args.train_dataset_type)(
            args, preprocess_val, is_train=False, epoch=epoch, tokenizer=tokenizer)
    if args.classification_dataset:
        data['rsicd'] = get_classification_dataset(args, preprocess_val, tokenizer=tokenizer)
    if args.retrieval_coco:
        data["retrieval_coco"] = get_retrieval_coco_dataset(args=args, preprocess_fn=preprocess_val,
                                                            tokenizer=tokenizer, output_dict=True)
    if args.retrieval_flickr:
        data["retrieval_flickr"] = get_retrieval_flickr_dataset(args=args, preprocess_fn=preprocess_val,
                                                                tokenizer=tokenizer, output_dict=True)

    return data



# The main dataset function for Itemized-cc0.3M, adapted from FLAIR's loader for cc3m-recap
def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    if is_train:
        input_shards = args.train_data
    else:
        input_shards = args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])


    external_captions = None
    if args.external_captions:
        ec = json.load(open(args.external_captions))
        external_captions = {int(k):ec[k] for k in ec}

    pipeline.extend([
        wds.select(filter_no_caption_or_no_image_json),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="json"),
        wds.map_dict(image=preprocess_img,
                     text=lambda text: sample_dict(text, k=args.num_sampled_captions, tokenizer=tokenizer,
                                                   sampling_mode=args.caption_sampling_mode,
                                                   pixelprose=args.pixelprose,
                                                   max_merged_num=args.max_merged_num,concat = args.concat_text_train if is_train else args.concat_text_val,
                                                   external_captions = external_captions,shuffle=is_train, padto = args.caption_pad_to_train if is_train else args.caption_pad_to_val)),
        wds.to_tuple("image", "text"),
        wds.map(reorder_tuple),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            print(num_shards)
            print(args.workers)
            print(args.world_size)
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil

        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def reorder_tuple(inp):
    a,b = inp
    c,d,e = b
    return a,c,d,e


# MSCOCO retrieval dataset
def read_coco_pairs(root_dir, dict_root_dir, split='train', sampling_mode=None, num_samples=None):
    """
    :param num_samples: int
    :param sampling_mode: str.
    :param root_dir: str; path to the dataset folder
    :param split: str; 'train' or 'val'
    :return: a list of dict: {'image_id': int, 'image': str, 'caption': str}
    """
    annotations_dir = os.path.join(root_dir, "annotations")
    if split == "train":
        captions_file = os.path.join(annotations_dir, "captions_train2017.json")
        images_dir = os.path.join(root_dir, "images", "train2017")
    else:
        split = 'val'
        captions_file = os.path.join(annotations_dir, "captions_val2017.json")
        images_dir = os.path.join(root_dir, "images", "val2017")

    with open(captions_file, 'r') as f:
        coco_data = json.load(f)

    image_id_to_path = {image['id']: os.path.join(images_dir, image['file_name']) for image in coco_data['images']}
    data_list = []
    cap_id = 0
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id in image_id_to_path:
            data_list.append({
                'image_id': image_id,
                'image': image_id_to_path[image_id],
                'caption': annotation['caption'],
                'caption_id': cap_id
            })
        cap_id += 1

    return data_list

# maps images to captions
def map_img_cap(data_list):
    """
    :param data_list: List of dict, each dict contains key 'image_id' and 'caption_id'
    :return: img2txt_dict, txt2img_dict
    """
    img2txt_dict = {}
    txt2img_dict = {}

    for entry in data_list:
        image_id = entry['image_id']
        caption_id = entry['caption_id']

        if image_id not in img2txt_dict:
            img2txt_dict[image_id] = [caption_id]
        else:
            img2txt_dict[image_id].append(caption_id)

        if caption_id not in txt2img_dict:
            txt2img_dict[caption_id] = [image_id]
        else:
            txt2img_dict[caption_id].append(image_id)
    return img2txt_dict, txt2img_dict


# Flickr retrieval dataset
def read_flickr_pairs(root_dir, split='train'):
    base_dir = os.path.dirname(root_dir)
    if split == 'train':
        captions_file = os.path.join(root_dir, "flickr30k_train.json")
    elif split == 'val':
        captions_file = os.path.join(root_dir, "flickr30k_val.json")
    else:
        captions_file = os.path.join(root_dir, "flickr30k_test.json")

    with open(captions_file, 'r') as f:
        flickr_data = json.load(f)

    data_list = []
    img_id, cap_id = 0, 0
    for annotation in flickr_data:
        image_path = os.path.join(base_dir, annotation['image'])
        caption_list = annotation["caption"]  # Now the caption should be a list
        for caption in caption_list:
            data_list.append({
                'image': image_path,
                'caption': caption,
                'image_id': img_id,
                'caption_id': cap_id
            })
            cap_id += 1
        img_id += 1
    return data_list



def pre_tokenize(tokenizer, data_list):
    for data in data_list:
        data["caption"] = tokenizer(data["caption"])
    return data_list


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
            self,
            urls,
            weights=None,
            nshards=sys.maxsize,
            worker_seed=None,
            deterministic=False,
            epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


class FlickrTextDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', tokenizer=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        #create data list
        logging.info(f"creating dataset list...")
        data_list = read_flickr_pairs(root_dir=self.root_dir, split=self.split)
        logging.info(f"dataset list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        if self.split == 'val':
            self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)
            logging.info(f"In validation mode, finish constructing the img_cap mapping dict for retrieval")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        caption = data["caption"].squeeze(dim=0)
        if self.split == 'val':
            cap_id = data["caption_id"]
            return caption, cap_id
        else:
            return caption


class FlickrImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from FlickrTextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


class COCOTextDataset(Dataset):
    '''
    Only loading captions and captions ID. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, dict_root_dir=None, transform=None, split='train', tokenizer=None, sampling_mode=None,
                 num_samples=None):
        self.root_dir = root_dir
        self.dict_root_dir = dict_root_dir
        self.transform = transform
        self.split = split
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        #create data list
        logging.info(f"creating dataset list...")
        data_list = read_coco_pairs(root_dir=self.root_dir, dict_root_dir=self.dict_root_dir, split=self.split,
                                    sampling_mode=self.sampling_mode, num_samples=self.num_samples)
        logging.info(f"dataset list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        if self.split == 'val':
            self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)
            logging.info(f"In validation mode, finish constructing the img_cap mapping dict for retrieval")
            #adding two dictionaries indicating the mapping between image index and text index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_id = data["image_id"]

        caption = data["caption"].squeeze(dim=0)

        if self.split == 'val':
            cap_id = data["caption_id"]
            return caption, cap_id  # Only retruning captions and cap_ids
        else:
            return caption

def extract_unique_img_list_from_data_list(data_list):
    """
    :param data_list: a list of dicts, each: {'image', 'image_id', 'caption', 'caption_id'}
    :return: img_list: a list of dicts, with all unique 'image' w.r.t. 'image_id'. So each new dict will be {'image', 'image_id'}
    """
    seen_ids = set()
    img_list = []

    for item in data_list:
        image_id = item['image_id']
        if image_id not in seen_ids:
            # Add to the list and mark the id as seen
            img_list.append({'image': item['image'], 'image_id': image_id})
            seen_ids.add(image_id)

    return img_list


class COCOImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from COCOTextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id

# Main function to get MSCOCO retrieval dataset
def get_retrieval_coco_dataset(args, preprocess_fn, tokenizer=None, output_dict=False):
    data_root_dir = args.coco_data_root_dir
    dict_root_dir = args.dict_root_dir
    split = 'val'
    sampler = None
    shuffle = False

    txt_dataset = COCOTextDataset(root_dir=data_root_dir, dict_root_dir=dict_root_dir, transform=preprocess_fn,
                                  split=split, tokenizer=tokenizer, sampling_mode=None, num_samples=None)
    img2txt_dict, txt2img_dict = txt_dataset.img2txt_dict, txt_dataset.txt2img_dict
    num_txt_samples = len(txt_dataset)

    img_dataset = COCOImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list, transform=preprocess_fn,
                                   split=split)
    num_img_samples = len(img_dataset)

    txt_dataloader = DataLoader(
        txt_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )

    txt_dataloader.num_samples = num_txt_samples
    txt_dataloader.num_batches = len(txt_dataloader)

    img_dataloader = DataLoader(
        img_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    img_dataloader.num_samples = num_img_samples
    img_dataloader.num_batches = len(img_dataloader)

    if output_dict:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler), img2txt_dict, txt2img_dict
    else:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler)

# Main function to get Flickr retrieval dataset
def get_retrieval_flickr_dataset(args, preprocess_fn, tokenizer=None, output_dict=False):
    data_root_dir = args.flickr_data_root_dir
    split = args.flickr_val_or_test
    sampler = None
    shuffle = False

    txt_dataset = FlickrTextDataset(root_dir=data_root_dir, transform=preprocess_fn,
                                    split=split, tokenizer=tokenizer)
    img2txt_dict, txt2img_dict = txt_dataset.img2txt_dict, txt_dataset.txt2img_dict
    num_txt_samples = len(txt_dataset)

    img_dataset = FlickrImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list, transform=preprocess_fn,
                                     split=split)
    num_img_samples = len(img_dataset)


    txt_dataloader = DataLoader(
        txt_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )

    txt_dataloader.num_samples = num_txt_samples
    txt_dataloader.num_batches = len(txt_dataloader)

    img_dataloader = DataLoader(
        img_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    img_dataloader.num_samples = num_img_samples
    img_dataloader.num_batches = len(img_dataloader)

    if output_dict:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler), img2txt_dict, txt2img_dict
    else:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler)

# Main function for RSICD dataset
def get_rsicd_dataset(args, preprocess_fn, is_train,epoch,tokenizer):
    split = 'train' if is_train else 'test'

    dataset = RSICD_Dataset(args=args, split=split, transform=preprocess_fn, tokenizer=tokenizer)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    num_samples = len(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,  # drop_last when training
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)

class RSICD_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train', transform=None, tokenizer=None, retraw = False):
        # retraw: whether to return raw image tensor for visualization purposes
        fname = args.rsicd_data_dir+'/'+split+'.csv'
        self.data = pd.read_csv(fname)
        self.transform = transform
        self.tokenizer = tokenizer
        self.args = args
        self.replacer = args.external_captions
        if self.replacer is not None:
            train,test = self.replacer.split(';')
            self.replacer = json.load(open(train if split == 'train' else test, 'r'))
        self.split = split
        self.retraw = retraw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # the rsicd images are stored as raw bytes in a csv. We have to extract them.
        item = self.data.iloc[idx]
        local_vars = {}
        exec('cc = ' + item.image, {}, local_vars)
        cc = local_vars['cc']
        image = Image.open(io.BytesIO(cc['bytes']))
        
        # Using original non-dedeuplicated captions
        if self.replacer is None:
            local_vars = {}
            exec('cc = ' + item.captions, {}, local_vars)
            cc = local_vars['cc']
            texts = [s.strip().lower() for s in cc[0].split('.')]
            if len(texts) > 6:
                print(cc)
        # use deduplicated captions from external file
        else:
            texts = self.replacer[str(idx)]
            texts = [s.strip().lower() for s in texts]

        if self.args.caption_sampling_mode == 'diverse_external': # with diverse sampling
            captions_list = texts
            n_captions = len(captions_list)
            k=self.args.num_sampled_captions
            sampled_sentences = []
            for _ in range(k):
                merged_num = random.randint(1, min(n_captions,self.args.max_merged_num))
                if merged_num == 1:
                    # Sample one caption
                    sampled_sentence = random.choice(captions_list)
                    sampled_sentences.append(sampled_sentence)
                else:
                    prob_flag = 0.5 # 50% merging subsequent captions, 50% merging captions from random positions
                    if random.random() < prob_flag:
                        sampled_sentence_list = random_sample_from_list(
                            captions_list, k=1, merged_num=merged_num)
                        sampled_sentences.extend(sampled_sentence_list)
                    else:
                        # Randomly select captions to merge
                        if n_captions >= merged_num:
                            captions_to_merge = random.sample(captions_list, merged_num)
                        else:
                            captions_to_merge = [random.choice(captions_list) for _ in range(merged_num)]
                        # Merge the captions
                        sampled_sentence = '. '.join(captions_to_merge)
                        sampled_sentences.append(sampled_sentence)
            texts = sampled_sentences
        #print(texts)


        if self.split == 'train':
            random.shuffle(texts)
        if self.retraw:
            raws = torchvision.transforms.PILToTensor()(image)
        if self.transform:
            image = self.transform(image)

        textmask = torch.zeros(self.args.num_sampled_captions)
        textmask[:len(texts)] = 1
        for j in range(self.args.num_sampled_captions - len(texts)): # add padding
            texts.append('placeholder')
        texts = self.tokenizer(texts)

        if self.retraw:
            idx = idx,raws

        return image, texts, textmask, idx

# RSICD dataset for classification only
class RSICD_class_Dataset(RSICD_Dataset):
    def __init__(self, args, transform=None, tokenizer=None, retraw = False):
        super().__init__(args, 'test', transform, tokenizer, retraw)
        self.idx2classname = json.load(open(args.rsicd_data_dir+'/large_label.json', 'r'))
    
    def __len__(self):
        return len(self.idx2classname)

    def __getitem__(self, idx):
        if self.retraw:
            return super().__getitem__(idx)
        image, texts, textmask, idx = super().__getitem__(idx)
        return image, self.idx2classname[idx]

# RSICD dataset for classification only
def get_classification_dataset(args, preprocess_fn, tokenizer=None):
    if args.classification_dataset == 'rsicd':
        dataset = RSICD_class_Dataset(args=args, transform=preprocess_fn, tokenizer=tokenizer)
    else:
        raise NotImplementedError
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = False

    num_samples = len(dataset)

    classes = ['airport','bareland','baseball field','beach','bridge','center','church','commercial','dense residential area','desert','farmland','forest','industrial area','meadow','medium residential area','mountain','park','parking','playground','pond','port','railway station','resort',
           'river','stadium','school','sparse residential area','square','storage tanks','viaduct']

    dataset.cls_texts = [['there is '+c for c in classes],['the '+c for c in classes]]

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,  # do not drop_last when testing
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)


    