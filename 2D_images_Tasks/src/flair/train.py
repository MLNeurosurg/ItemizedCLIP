"""
This code was adapted from FLAIR repository:
https://github.com/ExplainableML/flair/blob/main/src/flair/train.py

Which was adapted from OpenCLIP:
https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/train.py

The code integrates additional modifications and extensions to support the FLAIR models, and then included additional modifications to support ItemizedCLIP models.
Original authors: ML Foundations.
"""
import json,copy
import logging
import math
import os
import time
from unicodedata import normalize

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed import all_gather
from typing import Any, Dict, Optional, Tuple, Union

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def get_reordered_indices(batch_size, num_batches):
    """
    The original order: [(I_1, T_1), ..., (I_1, T_B), (I_2, T_1), ..., (I_2, T_B), ...
                          (T_1, T_B+1)...]
    reorder to [(I_1, T_1), ..., (I_1, T_N), (I_2, T_1), ..., (I_2, T_N), ... , (I_N, T_1), ..., I(I_N, T_N)]
    returning a list of reordered indices
    """
    reordered_indices = []
    for k in range(batch_size):
        for n in range(num_batches):
            base_idx = n * batch_size * batch_size
            img_idx_start = base_idx + k * batch_size
            img_idx_end = img_idx_start + batch_size
            reordered_indices.extend(list(range(img_idx_start, img_idx_end)))

    return reordered_indices

# train the model for 1 epoch
def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader

    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        
        images, texts, textmask, _ = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        textmask = textmask.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                losses = loss(**model_out, output_dict=True, textmask = textmask)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            raise NotImplementedError


        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()


        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                try:
                    val = val.item()
                except:
                    pass
                losses_m[key].update(val, batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {math.log(logit_scale_scalar):.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": math.log(logit_scale_scalar),
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})
            if 'aux_scale' in model_out:
                log_data.update({"aux_scale":math.log(model_out['aux_scale'].item()),"aux_bias":model_out['aux_bias'].item()})


            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

# Perform retrieval-based evaluations
def evaluate(model, data, epoch, args, loss = None, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    zero_shot_metrics = zero_shot_eval(
        model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        if 'retrieval_coco' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_coco']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_coco', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)
        if 'retrieval_flickr' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_flickr']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_flickr', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)
        if 'val' in data:
            data['val'].set_epoch(9)
            valloader = data['val'].dataloader
            if args.val_flair_retrieval_metrics:
                metrics = val_flair_retrieval_metrics('val',model,valloader,args,epoch,metrics,device,input_dtype,autocast)


    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if 'val' not in data:
        if 'retrieval_coco' in data:
            print(','.join([str(round(f*100 if f <= 1 else f,1)) for f in [metrics['retrieval_coco_text_to_image_R@1'],metrics['retrieval_coco_text_to_image_R@10'],metrics['retrieval_coco_text_to_image_R@100'],metrics['retrieval_coco_text_to_image_mean_rank'],metrics['retrieval_coco_image_to_text_R@1'],metrics['retrieval_coco_image_to_text_R@10'],metrics['retrieval_coco_image_to_text_R@100'],metrics['retrieval_coco_image_to_text_mean_rank']]]))
        if 'retrieval_flickr' in data:
            print(','.join([str(round(f*100 if f <= 1 else f,1)) for f in [metrics['retrieval_flickr_text_to_image_R@1'],metrics['retrieval_flickr_text_to_image_R@10'],metrics['retrieval_flickr_text_to_image_R@100'],metrics['retrieval_flickr_text_to_image_mean_rank'],metrics['retrieval_flickr_image_to_text_R@1'],metrics['retrieval_flickr_image_to_text_R@10'],metrics['retrieval_flickr_image_to_text_R@100'],metrics['retrieval_flickr_image_to_text_mean_rank']]]))

    log_data = {(name if 'flairval' in name else 'val/'+name): val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def remap_indices(merged_img_ids, cap_ids, img2txt_dict, txt2img_dict):
    """
    params:
    merged_img_ids: tensor of shape (M, D)
    cap_ids: tensor of shape (N) (But the ordering might be random)
    img2txt_dict: dict mapping each img_id to a list of cap_ids
    txt2img_dict: dict mappint each cap_id to an img_id (a list of one element)
    text_features: tensor of shape (N, D)
    """
    # so now ideally the cap_ids should be (0, ...N), so do the text_features
    # step2: re-index the merged_image_ids and re-do the mapping in the dict.
    # As the original image ids might just be random numbers, they don't represent the real ordering.

    img_id_mapping = {old_id.item(): new_idx for new_idx, old_id in enumerate(merged_img_ids)}
    reindexed_img_ids = torch.tensor([img_id_mapping[img_id.item()] for img_id in merged_img_ids])

    # Update the img2txt_dict and txt2img_dict with new indices
    new_img2txt_dict = {img_id_mapping[img_id]: [cap_id for cap_id in cap_id_list]
                        for img_id, cap_id_list in img2txt_dict.items()}

    new_txt2img_dict = {cap_id: img_id_mapping[txt2img_dict[cap_id][0]]
                        for cap_id in txt2img_dict.keys()}

    return new_img2txt_dict, new_txt2img_dict


def compute_retrieval(similarity_scores, txt2img, img2txt, withworst = False):
    """
    similarity_scpres: 2D matrix of similarity scores between images and texts (similarity_scores)
    txt2img: dict mapping each text index to its corresponding image index
    img2txt: dict mapping each image index to a list of corresponding text indices
    withworst: whether to compute top-k retrieval on the positive text item with worst similarity for each image
    """
    if isinstance(similarity_scores, tuple):
        i2t_similarity_score, t2i_similarity_score = similarity_scores
    else:
        # Otherwise, treat similarity_scores as a single matrix for t2i
        t2i_similarity_score = similarity_scores.t()
        i2t_similarity_score = similarity_scores

    t2i_ranks = torch.zeros(t2i_similarity_score.shape[0])

    for index, score in enumerate(t2i_similarity_score):
        inds = torch.argsort(score, descending=True)
        t2i_ranks[index] = torch.where(inds == txt2img[index])[0][0]


    # Compute metrics
    tr1 = len(torch.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
    tr5 = len(torch.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
    tr10 = len(torch.where(t2i_ranks < 10)[0]) / len(t2i_ranks)
    tr50 = len(torch.where(t2i_ranks < 50)[0]) / len(t2i_ranks)
    tr100 = len(torch.where(t2i_ranks < 100)[0]) / len(t2i_ranks)
    t2i_report_dict = {
        "text_to_image_R@1": tr1,
        "text_to_image_R@5": tr5,
        "text_to_image_R@10": tr10,
        "text_to_image_R@50": tr50,
        "text_to_image_R@100": tr100,
        "text_to_image_mean_rank": t2i_ranks.mean().item() + 1,
        "text_to_image_median_rank": np.floor(np.median(t2i_ranks.numpy())) + 1
    }

    # comput image -> text
    i2t_ranks = torch.zeros(i2t_similarity_score.shape[0])
    i2t_worstranks = torch.zeros_like(i2t_ranks)
    for index, score in enumerate(i2t_similarity_score):
        inds = torch.argsort(score, descending=True)
        # Score
        rank = 1e10
        worstrank = -100
        for i in img2txt[index]:
            tmp = torch.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
            if tmp > worstrank:
                worstrank = tmp
        i2t_ranks[index] = rank
        i2t_worstranks[index] = worstrank

    # Compute metrics
    ir1 = len(torch.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
    ir5 = len(torch.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
    ir10 = len(torch.where(i2t_ranks < 10)[0]) / len(i2t_ranks)
    ir50 = len(torch.where(i2t_ranks < 50)[0]) / len(i2t_ranks)
    ir100 = len(torch.where(i2t_ranks < 100)[0]) / len(i2t_ranks)

    i2t_report_dict = {
        "image_to_text_R@1": ir1,
        "image_to_text_R@5": ir5,
        "image_to_text_R@10": ir10,
        "image_to_text_R@50": ir50,
        "image_to_text_R@100": ir100,
        "image_to_text_mean_rank": i2t_ranks.mean().item() + 1,
        "image_to_text_median_rank": np.floor(np.median(i2t_ranks.numpy())) + 1
    }

    if withworst:
        # Compute worstrank metrics
        ir1w = len(torch.where(i2t_worstranks < 1)[0]) / len(i2t_worstranks)
        ir5w = len(torch.where(i2t_worstranks < 5)[0]) / len(i2t_worstranks)
        ir10w = len(torch.where(i2t_worstranks < 10)[0]) / len(i2t_worstranks)
        ir50w = len(torch.where(i2t_worstranks < 50)[0]) / len(i2t_worstranks)
        ir100w = len(torch.where(i2t_worstranks < 100)[0]) / len(i2t_worstranks)

        i2t_report_dict_worst = {
            "image_to_text_R@1_worst": ir1w,
            "image_to_text_R@5_worst": ir5w,
            "image_to_text_R@10_worst": ir10w,
            "image_to_text_R@50_worst": ir50w,
            "image_to_text_R@100_worst": ir100w,
            "image_to_text_mean_rank_worst": i2t_worstranks.mean().item() + 1,
            "image_to_text_median_rank_worst": np.floor(np.median(i2t_worstranks.numpy())) + 1
        }
    else:
        i2t_report_dict_worst = {}


    metrics = {**t2i_report_dict, **i2t_report_dict, **i2t_report_dict_worst}
    return metrics

# Use flair zero-shot retrieval (i.e. TCSim-based retrieval) on val set
def val_flair_retrieval_metrics(keyword, model, dataloader, args, epoch, metrics, device,
                       input_dtype, autocast):
    vmetrics = {}
    keyword = keyword+'_'
    lossmetermeanrank = AverageMeter()
    lossmetermaxrank = AverageMeter()
    lossmetert2i1 = AverageMeter()
    lossmetert2i10 = AverageMeter()
    lossmetert2i100 = AverageMeter()
    lossmeteri2t1 = AverageMeter()
    lossmeteri2t10 = AverageMeter()
    lossmeteri2t100 = AverageMeter()
    lossmeteri2t1000all = AverageMeter()
    lossmeteri2t10all = AverageMeter()
    lossmeteri2t100all = AverageMeter()
    umodel = unwrap_model(model)
    with torch.no_grad():
        all_texts = []
        all_image_tokens = []
        idxmap = []
        currind = 0

        # encode image and text with our model
        for i,batch in tqdm(enumerate(dataloader)):
            images, texts, textmask, _ = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts[:,0:args.num_sampled_captions,:].to(device=device, non_blocking=True)
            textmask = textmask[:,0:args.num_sampled_captions].to(device=device, non_blocking=True)
            with autocast():
                model_out = umodel(images, texts)
                indices = textmask.nonzero(as_tuple=False)[:,0] + currind
                boolmask = textmask.view(-1).bool()
                embdim = model_out['text_features'].size()[-1]
                all_texts.append(model_out['text_features'].view(-1,embdim)[boolmask])
                all_image_tokens.append(model_out['image_tokens'])
                idxmap.append(indices)
                currind += len(textmask)
        # concatenate and save all embedding
        all_texts = torch.cat(all_texts,dim=0)
        all_image_tokens = torch.cat(all_image_tokens,dim=0)
        idxmap = torch.cat(idxmap,dim=0)
        step_size=100
        step_size_2 = 80

        # Perform TCSim over all possible image-text-item pairs, and store their TCSim in all_sims
        all_sims = torch.zeros(len(all_image_tokens),len(all_texts)).to(all_texts.device)
        for i in tqdm(range(len(all_texts)//step_size+1)):
            if step_size * i + step_size > len(all_texts):
                txtbatch = all_texts[step_size*i:]
            else:
                txtbatch = all_texts[step_size*i:step_size*i+step_size] # step_size x dim
            for j in range(len(all_image_tokens)//step_size_2+1):
                if step_size_2 * j + step_size_2 > len(all_image_tokens):
                    imgbatch = all_image_tokens[step_size_2*j:]
                else:
                    imgbatch = all_image_tokens[step_size_2*j:step_size_2*j+step_size_2]
                with autocast():
                    outembs = model_out['visual_proj'](txtbatch.unsqueeze(0).expand(len(imgbatch),-1,-1),imgbatch,imgbatch) # output should be step_size_2 x step_size x d
                    outembs = F.normalize(outembs,dim=-1)
                    inembs = F.normalize(txtbatch,dim=-1)
                    sims = torch.einsum('bk,abk->ab',inembs,outembs) # step_size_2 x step_size
                    all_sims[j*step_size_2:j*step_size_2+len(sims),i*step_size:i*step_size+len(sims[0])] = sims

        # calculate metrics based on all_sims
        for index, score in enumerate(all_sims):
            inds = torch.argsort(score, descending=True)
            # Score
            corrects = torch.where(idxmap == index)[0]
            ranks = torch.zeros_like(corrects).float()
            for idx,i in enumerate(corrects):
                tmp = torch.where(inds == i)[0][0]
                ranks[idx] = tmp
            lossmetermeanrank.update(ranks.mean().item())
            lossmetermaxrank.update(ranks.max().item())
            
            lossmeteri2t1.update(1 if ranks.min() < 1 else 0)
            lossmeteri2t10.update(1 if ranks.min() < 10 else 0)
            lossmeteri2t100.update(1 if ranks.min() < 100 else 0)

            lossmeteri2t10all.update(1 if ranks.max() < 10 else 0)
            lossmeteri2t100all.update(1 if ranks.max() < 100 else 0)
            lossmeteri2t1000all.update(1 if ranks.max() < 1000 else 0)


        for index, score in enumerate(all_sims.t()):
            inds = torch.argsort(score, descending=True)
            correct = idxmap[index]
            rank = torch.where(inds == correct)[0].item()
            lossmetert2i1.update(1 if rank < 1 else 0)
            lossmetert2i10.update(1 if rank < 10 else 0)
            lossmetert2i100.update(1 if rank < 100 else 0)
        
        metrics.update({'flairval/Flair_Rank_All_Entry': lossmetermeanrank.avg,'flairval/Flair_Rank_Worst_Entry':lossmetermaxrank.avg})
        metrics.update({'flairval/FlairRankT2i@1': lossmetert2i1.avg,'flairval/FlairRankT2I@10': lossmetert2i10.avg,'flairval/FlairRankT2I@100': lossmetert2i100.avg,
                'flairval/FlairRankI2T@1': lossmeteri2t1.avg,'flairval/FlairRankI2T@10': lossmeteri2t10.avg,'flairval/FlairRankI2T@100': lossmeteri2t100.avg,
                'flairval/FlairRankI2T_COMPLETE@10': lossmeteri2t10all.avg,'flairval/FlairRankI2T_COMPLETE@100': lossmeteri2t100all.avg,'flairval/FlairRankI2T_COMPLETE@1000': lossmeteri2t1000all.avg})
        return metrics
        

        

# compute retrieval metrics on mscoco/flickr
def retrieval_on_split(keyword, model, txt_loader, img_loader, img2txt_dict, txt2img_dict, args, epoch, metrics, device,
                       input_dtype, autocast):
    num_txt_samples = txt_loader.num_samples
    num_img_samples = img_loader.num_samples
    all_image_features, all_text_tokens, all_text_features = [], [], []
    all_local_text_tokens = []
    all_img_ids, all_cap_ids = [], []

    with torch.no_grad():
        # first loop over the text dataloader to store all text embeddings
        #for i, batch in tqdm(enumerate(txt_loader), total=len(txt_loader), desc="Processing Texts"):
        for i, batch in tqdm(enumerate(txt_loader)):
            texts, cap_id = batch
            texts = texts.to(device=device, non_blocking=True)
            with autocast():
                if args.inference_with_flair:
                    global_text_token, local_text_tokens = unwrap_model(model).encode_text(texts, normalize=False)
                    global_text_token, local_text_tokens = unwrap_model(model).text_post(
                        global_text_token), unwrap_model(model).text_post(local_text_tokens)
                    text_features = F.normalize(global_text_token, dim=-1)
                    all_text_tokens.append(global_text_token.squeeze(1))  # GPU
                    all_local_text_tokens.append(local_text_tokens)  # GPU
                else:
                    text_features = unwrap_model(model).encode_text(texts, normalize=False)
                    if isinstance(text_features,tuple):
                        text_features = unwrap_model(model).text_post(text_features[0])
                    if unwrap_model(model).txt_proj is not None:
                        umodel = unwrap_model(model)
                        text_features = get_txt_projs_out({'image_features':torch.zeros(len(text_features)),'text_features':text_features,'txtq':None,'txtproj':None},torch.ones(len(text_features),1).to(text_features.device))

                    text_features = F.normalize(text_features, dim=-1)

                all_text_features.append(text_features.detach().cpu())  # cpu list of N, each of shape (B, D)
                all_cap_ids.append(cap_id.detach().cpu())
        all_text_features_tensor = torch.cat(all_text_features)  # (N, 512)
        cap_ids = torch.cat(all_cap_ids)

        # Now loop over the images to compute similarity scores
        if args.inference_with_flair:
            mode = "inference_with_flair"
            all_text_tokens_tensor = torch.cat(all_text_tokens)  # on GPU
            all_local_text_tokens_tensor = torch.cat(all_local_text_tokens)

            similarity_scores, img_ids = compute_similarity_scores_attn_pool(
                model, img_loader, all_text_features_tensor, all_text_tokens_tensor, device, input_dtype, autocast, mode
            )
        else:
            similarity_scores, img_ids = compute_similarity_scores_original_clip(model, img_loader,
                                                                                 all_text_features_tensor, device,
                                                                                 input_dtype,
                                                                                 autocast,
                                                                                 mode='original_clip')
        new_img2txt_dict, new_txt2img_dict = remap_indices(merged_img_ids=img_ids, cap_ids=cap_ids,
                                                           img2txt_dict=img2txt_dict, txt2img_dict=txt2img_dict)

        retrieval_metrics = compute_retrieval(similarity_scores=similarity_scores,
                                              txt2img=new_txt2img_dict,
                                              img2txt=new_img2txt_dict)

        if keyword != '':
            temp_retrieval_metrics = {}
            keyword = keyword + '_'
            for k, v in retrieval_metrics.items():
                temp_retrieval_metrics[keyword + k] = v
            retrieval_metrics = temp_retrieval_metrics

        if "epoch" in metrics:  # we only need one epoch information
            metrics.update(
                {**retrieval_metrics,
                 f"{keyword}num_text_samples": num_txt_samples,
                 f"{keyword}num_image_samples": num_img_samples
                 }
            )
        else:
            metrics.update(
                {**retrieval_metrics,
                 f"epoch": epoch,
                 f"{keyword}num_text_samples": num_txt_samples,
                 f"{keyword}num_image_samples": num_img_samples
                 }
            )

    return metrics

# computing similarity scores with original CLIP zero-shot retrieval method 
def compute_similarity_scores_original_clip(model, img_loader, all_text_features_tensor, device, input_dtype,
                                            autocast, mode='original_clip'):
    all_image_features = []
    all_img_ids = []

    for i, batch in enumerate(img_loader):
        images, img_id = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        all_img_ids.append(img_id.detach().cpu())
     
        with autocast():
            if mode == 'original_clip':
                image_features = unwrap_model(model).encode_image(images, normalize=True)
                if isinstance(image_features,tuple):
                    image_features =unwrap_model(model).image_post(image_features[0])
            elif mode == 'imgcon':
                _, local_image_tokens = unwrap_model(model).encode_image(images)
                local_image_tokens = unwrap_model(model).image_post(local_image_tokens)
                image_features = unwrap_model(model).visual_proj(local_image_tokens.mean(dim=1, keepdim=True), local_image_tokens, local_image_tokens)
                image_features = image_features.squeeze(1)
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = unwrap_model(model).logit_scale.exp()
            all_image_features.append(image_features.detach().cpu())

    all_image_features_tensor = torch.cat(all_image_features)
    img_ids = torch.cat(all_img_ids)

    similarity_scores = logit_scale.cpu() * all_image_features_tensor.float() @ all_text_features_tensor.float().t()
    return similarity_scores, img_ids

# compute similarity scores with FLAIR's zero-shot retrieval method (i.e. TCSim-based retrieval)
def compute_similarity_scores_attn_pool(model, img_loader, all_text_features_tensor, all_text_tokens_tensor, device,
                                        input_dtype,
                                        autocast, mode):
    logits_per_image_list = []
    all_img_ids = []

    for i, batch in enumerate(img_loader):
        images, img_id = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        all_img_ids.append(img_id.detach().cpu())
        with autocast():
            if mode == 'inference_with_flair':
                _, image_embeddings = unwrap_model(model).encode_image(images, normalize=False)
                image_embeddings = unwrap_model(model).image_post(image_embeddings) 
                img_features_after_conditioning = unwrap_model(model).visual_proj(
                    all_text_tokens_tensor.unsqueeze(0),
                    image_embeddings,
                    image_embeddings
                )
                img_features_after_conditioning = F.normalize(img_features_after_conditioning, dim=-1).detach().cpu()
                embed_dim = img_features_after_conditioning.shape[-1]
                img_features_after_conditioning = img_features_after_conditioning.contiguous().view(-1, embed_dim)
            else:
                embed_dim = all_text_features_tensor.shape[-1]
                img_features_after_conditioning = unwrap_model(model).visual_proj(
                    all_text_tokens_tensor.unsqueeze(0),
                    image_embeddings,
                    image_embeddings
                ).detach().cpu().contiguous().view(-1, embed_dim)

            logit_scale = unwrap_model(model).logit_scale.exp()
            logits_per_image = (logit_scale.cpu() * torch.einsum('ij,ij->i', img_features_after_conditioning,
                                                                 all_text_features_tensor)).unsqueeze(0).detach().cpu()
        logits_per_image_list.append(logits_per_image)

    img_ids = torch.cat(all_img_ids)  # shape (M)
    similarity_scores = torch.cat(logits_per_image_list)  # shape (M, N)
    return similarity_scores, img_ids
