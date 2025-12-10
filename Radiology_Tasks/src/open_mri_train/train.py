"""
Adapted from train.py from HLIP repository: https://github.com/Zch0414/hlip/blob/master/src/hlip_train/train.py
"""

import sys
sys.path.append('.')
sys.path.append('..')

import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch import distributed as dist

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from open_ct_rate.zeroshot_ct_rate import zero_shot


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

# Main training loop for 1 epoch
def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // (args.accum_freq * args.accum_batch)
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_negatives, accum_features = [], [], [], {}
    images, texts, negatives, serienames, numsubreports, textmasks = [], [], [], [], [],[]


    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, mini_batch in enumerate(dataloader):
        i_accum = i // (args.accum_freq * args.accum_batch)
        step = num_batches_per_epoch * epoch + i_accum
        
        if not args.skip_scheduler:
            scheduler(step)

        _images, _texts, _negatives, _serienames = mini_batch
        if args.itemizedclip_text or args.use_itemizedclip_loss:
            _texts, _textmask = _texts
            textmasks.append(_textmask.to(device=device, non_blocking=True))                
        images.append(_images.to(device=device, dtype=input_dtype, non_blocking=True))
        texts.append(_texts.to(device=device, non_blocking=True))
        negatives.append(_negatives.to(device=device, non_blocking=True))

        if args.use_serienames:
            serienames.append(_serienames.to(device=device, non_blocking=True))

        # Accumulate batches
        if ((i + 1) % args.accum_batch) > 0:
            continue


        
        optimizer.zero_grad()

            
        images = torch.cat(images, dim=0); texts = torch.cat(texts, dim=0); negatives = torch.cat(negatives, dim=0)
        visualinput = images

        if args.itemizedclip_text or args.use_itemizedclip_loss:
            textmasks = torch.cat(textmasks,dim=0)
            if args.itemizedclip_text:
                texts = (texts,textmasks)
        
        if args.use_serienames:
            serienames = torch.cat(serienames, dim=0)
            visualinput = (images,serienames)

        data_time_m.update(time.time() - end)

        if args.accum_freq == 1:
            with autocast():
                model_out = model(visualinput, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                if args.custom_clip_loss:
                    losses = loss(**model_out, false_negatives=negatives.float() if args.correct_false_negatives else None, output_dict=True)
                elif args.use_itemizedclip_loss:
                    losses = loss(**model_out,textmask=textmasks, output_dict=True, normals = negatives)
                else:
                    losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                lossname = 'loss'
                losses[lossname] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
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

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_negatives, accum_features = [], [], [], {}

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
                losses_m[key].update(val.item(), batch_size)

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
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

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
        
        # reset batch accum
        images, texts, negatives, serienames,numsubreports, textmasks = [], [], [], [],[],[]
        
        
    # end for



def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None, val2 = False, loss = None, keyword='val'):
    metrics = {}
    if (not is_master(args)) and ((not args.use_itemizedclip_loss) or (args.zeroshot_ct_rate)):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        if val2:
            dataloader = data['val2'].dataloader
        else:
            dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        if args.use_itemizedclip_loss:
            
            lossmeters = {}
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
            with torch.inference_mode():
                cum_textmasks = []
                cum_image_features = []
                cum_text_features = []
                cum_image_tokens = []
                for i, batch in enumerate(dataloader):
                    images, texts, negatives, serienames = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts,textmasks = texts
                    
                    
                    textmasks = textmasks.to(device=device,non_blocking=True)
                    negatives = negatives.to(device=device,non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    cum_textmasks.append(textmasks)

                    with autocast():
                        model_out = model((images,serienames), texts)
                        cum_image_features.append(model_out['image_features'])
                        cum_image_tokens.append(model_out['image_tokens'])
                        cum_text_features.append(model_out['text_features'])
                        #losses = loss(**model_out,textmask=textmasks, output_dict=True, normals = negatives,localloss=True)
                        #for k in losses:
                        #    if keyword+k not in lossmeters:
                        #        lossmeters[keyword+k] = AverageMeter()
                        #    lossmeters[keyword+k].update(losses[k].item(),n=len(images))
                #cum_image_features = torch.cat(cum_image_features,dim=0)
                cum_text_features = torch.cat(cum_text_features, dim=0)
                #cum_image_tokens = torch.cat(cum_image_tokens, dim=0)
                cum_textmasks = torch.cat(cum_textmasks, dim=0)
                all_text_features = [torch.zeros_like(cum_text_features) for _ in range(args.world_size)]
                all_textmasks = [torch.zeros_like(cum_textmasks) for _ in range(args.world_size)]
                

                dist.all_gather(all_text_features,cum_text_features) 
                all_text_features = torch.cat(all_text_features, dim=0)
                dist.all_gather(all_textmasks,cum_textmasks)
                all_textmasks = torch.cat(all_textmasks,dim=0) # 1000x7
                all_text_features = F.normalize(all_text_features.view(-1,all_text_features.shape[-1]), dim=-1)
                sims = torch.zeros(len(cum_image_tokens),len(all_text_features)).to(device)
                for enu, image_tokens in enumerate(cum_image_tokens):
                    with autocast():
                        attended_txt_feats = model_out['visual_proj'](all_text_features.view(1,-1,all_text_features.shape[-1]),image_tokens,image_tokens)[0] 
                    attended_txt_feats = F.normalize(attended_txt_feats,dim=-1)
                    sim = torch.einsum('ab,ab->a',all_text_features,attended_txt_feats)
                    sims[enu] = sim
                all_sims = [torch.zeros_like(sims) for _ in range(args.world_size)]
                dist.all_gather(all_sims,sims)
                all_sims = torch.cat(all_sims,dim=0) # 1000x7000

                if not is_master(args):
                    return metrics

                simmask = all_textmasks.view(-1).bool()
                maps = torch.arange(len(all_textmasks)).unsqueeze(1).expand(-1,7).to(device) # 7000 in size
                all_sims = all_sims[:,simmask]
                idxmap = maps.reshape(-1)[simmask]

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


                metrics.update({'TCSimval/TCSim_Rank_All_Entry': lossmetermeanrank.avg,'TCSimval/TCSim_Rank_Worst_Entry':lossmetermaxrank.avg})
                metrics.update({'TCSimval/TCSimRankT2i@1': lossmetert2i1.avg,'TCSimval/TCSimRankT2I@10': lossmetert2i10.avg,'TCSimval/TCSimRankT2I@100': lossmetert2i100.avg,
                        'TCSimval/TCSimRankI2T@1': lossmeteri2t1.avg,'TCSimval/TCSimRankI2T@10': lossmeteri2t10.avg,'TCSimval/TCSimRankI2T@100': lossmeteri2t100.avg,
                        'TCSimval/TCSimRankI2T_COMPLETE@10': lossmeteri2t10all.avg,'TCSimval/TCSimRankI2T_COMPLETE@100': lossmeteri2t100all.avg,'TCSimval/TCSimRankI2T_COMPLETE@1000': lossmeteri2t1000all.avg})
                 
                for k in lossmeters:
                    metrics[k] = lossmeters[k].avg
                
                
        # Evaluate with normal CLIP cosine similarity instead of TCSim
        else:
            # FIXME this does not scale past small eval datasets
            # all_image_features @ all_text_features will blow up memory and compute very quickly
            cumulative_clip_score = 0.0
            cumulative_gen_loss = 0.0
            all_image_features, all_text_features = [], []
            all_negatives = []
            with torch.inference_mode():
                for i, batch in enumerate(dataloader):
                    images, texts, negatives, serienames = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    if args.itemizedclip_text:
                        texts = (texts[0].to(device=device, non_blocking=True),texts[1].to(device=device, non_blocking=True))
                    else:
                        texts = texts.to(device=device, non_blocking=True)
                    all_negatives.append(negatives.cpu())
                    
                    with autocast():
                        model_out = model((images,serienames), texts)
                        image_features = model_out["image_features"]
                        text_features = model_out["text_features"]
                        if args.itemizedclip_text:
                            text_features, _ = text_features
                        logit_scale = model_out["logit_scale"]
                        # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                        # however, system RAM is easily exceeded and compute time becomes problematic
                        all_image_features.append(image_features.cpu())
                        all_text_features.append(text_features.cpu())
                        logit_scale = logit_scale.mean()

                        # compute clip score
                        clip_scores_per_image = torch.clamp(image_features @ text_features.t(), min=0) * 100
                        total_clip_scores = clip_scores_per_image.trace()
                        batch_size = images.shape[0]

                        gen_loss = maybe_compute_generative_loss(model_out)

                    cumulative_clip_score += total_clip_scores * batch_size
                    num_samples += batch_size
                    if is_master(args) and (i % 100) == 0:
                        logging.info(
                            f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                            f"Clip Score: {cumulative_clip_score / num_samples:.6f}\t"
                        )


                val_metrics = get_clip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                    negatives=torch.cat(all_negatives)
                )
                clip_score = cumulative_clip_score / num_samples
                metrics.update(
                    {**val_metrics, "clip_val_score":clip_score.item(), "epoch": epoch, "num_samples": num_samples}
                )
                if gen_loss is not None:
                    gen_loss = cumulative_gen_loss / num_samples
                    metrics.update({"val_generative_loss": gen_loss.item()})
    if 'zeroshot-ct-rate' in data:
        res,_ = zero_shot(model,tokenizer,data['zeroshot-ct-rate'].dataloader,args)
        metrics.update(res['* mean'])        

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )
    if val2:
        log_data = {"val2/" + name: val for name, val in metrics.items()}
    else:
        log_data = {"val/" + name: val for name, val in metrics.items()}

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
            num_batches_per_epoch = dataloader.num_batches // (args.accum_freq * args.accum_batch)
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics

# Get CLIP evaluation metrics from image and text features
def get_clip_metrics(image_features, text_features, logit_scale, negatives):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        all_preds = (ranking == ground_truth).float()
        negative_preds = negatives.squeeze()[ranking]
        preds = torch.where(negatives.bool(), negative_preds, all_preds)
        preds = preds.float().argmax(dim=1)  # shape: [bs]
        preds = preds.cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

