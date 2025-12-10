import glob
import logging
import os
import re
import subprocess
import sys
import random
import json
from datetime import datetime
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.distributed import get_rank
from huggingface_hub import hf_hub_download

sys.path.append('.')
sys.path.append('..')

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync

from open_clip_train.precision import get_autocast

from flair.params import parse_args
from flair.factory import create_model_and_transforms, get_tokenizer, create_loss
from flair.train import train_one_epoch, evaluate
from flair.data import get_data
from visualize.visualizer import visualize_itemizedclip

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def download_weights_from_hf(model_repo, filename):
    # Define the custom cache directory relative to the current script
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    local_path = hf_hub_download(repo_id=model_repo, filename=filename, cache_dir=cache_dir)
    return local_path


def main(args):
    args = parse_args(args)
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)



    resume_latest = args.resume == 'latest'
    

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    model_kwargs = {'args': args}

    # based on huggingface model name, download the pre-trained weights, the downloaded path is passed as the 'pretrained' arguments
    if args.huggingface_model_name != '':
        huggingface_model_name, huggingface_repo_name = args.huggingface_model_name, args.huggingface_repo_name
        args.pretrained = download_weights_from_hf(model_repo=huggingface_repo_name, filename=huggingface_model_name)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,

        **model_kwargs
    )
    model.to(device)
    random_seed(args.seed, args.rank)


    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")


    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")


    # initialize datasets
    tokenizer = get_tokenizer(args.model)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)


    if 'train' not in data:
        # Evaluate.
        autocast = get_autocast(args.precision)
        visualize_itemizedclip(model, data, args, autocast)
        return


if __name__ == "__main__":
    main(sys.argv[1:])
