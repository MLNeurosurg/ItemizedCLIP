"""
Adapted from FLAIR loss function implementation: https://github.com/ExplainableML/flair/blob/main/src/flair/loss.py

In addition to TCS and MPS, added the new IIS and KTA objectives, and modified TCS to ILA loss
"""



import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models import checkpoint
import math

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# Gather features from all GPUs for loss calculation
def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features



# single direction neighbor exchange. Sends tensor to to_rank and receives tensor from from_rank.
def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv

# bidirectional neighbor exchange. Sends tensor_to_left to left_rank and tensor_to_right to right_rank, and receives tensors from both sides.
def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


# generates target map for MPS loss. Labels which text/image pairs are positive (1) or negative (-1)
def get_multi_positive_mps(target, k):
    """
    :param target: tensor of shape (b, b*k), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b*k), for each row i, the col [i*k, (i+1)*k] should be ones
    """
    for i in range(target.shape[0]):
        target[i, i * k:(i + 1) * k] = 1
    return target


# generates target map for ILA loss. Labels which text/image pairs are positive (1) or negative (-1)
def get_multi_positive_ila(target, k):
    """
    :param target: tensor of shape (b, b+k-1), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b+k-1), for each row i, the col [i, i+k) should be ones
    """
    for i in range(target.shape[0]):
        target[i, i: i + k] = 1
    return target



# MPS logit calculation
def get_mps_logits(image_features, text_features, logit_scale, logit_bias=None):
    logits = logit_scale * image_features @ text_features.T  # if multi-cap: (B, B*K)
    if logit_bias is not None:
        logits += logit_bias
    return logits

# Get ground truth labels (positive/negative) for MPS loss.
# The negative-only flag indicates whether to only consider negative pairs (for distributed training, all non-local pairs are negative)
def get_mps_ground_truth(device, dtype, target_shape, negative_only=False,
                                        num_captions=4, textmask = None):
    dim0, dim1 = target_shape  # (B, B*K)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B*K)
    if not negative_only:
        labels = get_multi_positive_mps(target=labels, k=num_captions)
    if textmask is not None:
        gtmask = textmask[:,0:num_captions].reshape(1,dim1)
        return labels,gtmask
    return labels, None


# Calculate logits for IIS loss
def get_intra_logits(image_features, text_features, logit_scale, logit_bias=None):
    """
    image_features: (B, K, D),
    text_features: (B, K, D).
    Target: (B, K, K)
    """
    logits = logit_scale * torch.einsum('bkd,bjd->bkj', image_features, text_features)
    # logits = logit_scale * image_features @ text_features.T  
    if logit_bias is not None:
        logits += logit_bias
    return logits


# Get ground truth labels (positive/negative) for ILA loss.
# The negative-only flag indicates whether to only consider negative pairs (for distributed training, all non-local pairs are negative)
def get_ila_ground_truth(device, dtype, target_shape, negative_only=False, num_captions=4, textmask = None):
    dim0, dim1 = target_shape  # (B, B+K-1)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B+K-1)
    if not negative_only:
        labels = get_multi_positive_ila(target=labels, k=num_captions)
    return labels


# calculate logits for ILA loss (Basically computes TCSim)
# features 0 and 1 are text conditioned image features and text features, respectively (order interchangeable)
def get_ila_logits(features_0, features_1, logit_scale, logit_bias=None):
    logits = logit_scale * torch.einsum('bij,bij->bi', features_0, features_1)
    if logit_bias is not None:
        logits += logit_bias
    return logits


class ItemizedCLIPLoss(nn.Module):

    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
            num_cap_per_img=8, # maximum number of text items per study
            added_mps_loss=False, # whether to add mps loss
            iis_loss = None, # weight of iis loss. None means no iis loss
            mps_fac = 0.5, # weight of mps loss
            maskposrate=1.0, # The masking rate for ILA masking. The larger this value is, the less masking there is (1.0 is no masking, 0.9 means 10% chance masking each local image feature for each head)
            ila_fac = 0.5, # weight of ila loss
            no_normal_check = False, # whether to skip normal check or not
            comp_upfac = 1.0, # the upweighting factor for worst positives (UWP). 1.0 means no upweighting, 2.0 means doubling the loss over the worst positive logit, and so on
            key_token_alignment_loss = None, # weight of key token alignment loss. None means no KTA loss
            key_token_thresh = 0.1 # threshold for key token selection in KTA loss
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}
        self.num_cap_per_img = num_cap_per_img
        self.added_mps_loss = added_mps_loss
        self.iis_loss = iis_loss
        self.mps_fac = mps_fac
        self.maskposrate = maskposrate
        self.ila_fac = ila_fac
        self.no_normal_check = no_normal_check

        self.comp_upfac = comp_upfac
        self.key_token_alignment_loss = key_token_alignment_loss
        self.key_token_thresh = key_token_thresh




    def _loss_with_attn_pool(self, image_features, image_tokens, text_features, logit_scale,
                             logit_bias=None, negative_only=False, visual_proj=None, g_text_features=None, textmask = None, show_all_loss = False, normals1 = None, normals2 = None):
        """
        image_features: (B, D) global image features
        image_tokens: (B, L, D) local image tokens
        text_features: (B, K. D) text features (K=num_cap_per_img)
        logit_scale: scaling factor for logits
        logit_bias: optional bias for logits
        negative_only: whether all text-image pairs are negative (for distributed training)
        visual_proj: visual projection module
        g_text_features: same as text features (legacy from FLAIR code)
        textmask: since ItemizedCLIP allows varying amount of items per image, this masks which text items are real (1) and which ones are just padding (0)
        show_all_loss: whether to show all loss components in returned dictionary
        normals1: normal labels for images
        normals2: normal labels for text items
        """

        
        bxbmask = torch.outer(normals1.long(),normals2.long())
        if not negative_only:
            bxbmask.fill_diagonal_(0)

        all_losses = {}

        num_heads = visual_proj.n_head

        total_text_per_image = text_features.shape[1]

        # Generate the ILA masking
        imgmask = torch.bernoulli((torch.ones_like(image_tokens[:,:,0])*self.maskposrate).unsqueeze(1).unsqueeze(1).expand(-1,num_heads,total_text_per_image,-1)).view(len(image_tokens)*num_heads,total_text_per_image,-1)

        # Calculate the text-conditioned image features
        local_image_features = checkpoint(visual_proj,text_features, image_tokens, image_tokens, mask=imgmask)  # (B, B+K-1, D)

        local_image_features = F.normalize(local_image_features, dim=-1)
        global_text_features = F.normalize(text_features, dim=-1)

        i2t_logits = get_ila_logits(local_image_features, global_text_features, logit_scale, logit_bias)

        i2t_labels = get_ila_ground_truth(device=text_features.device,
                                        dtype=text_features.dtype,
                                        target_shape=i2t_logits.size(),
                                        negative_only=negative_only,
                                        num_captions=self.num_cap_per_img)

        # compute text masks and normal-check masks to ensure no loss is calculated over padding text items, and normal text and normal image are never used as negative pairs
        with torch.no_grad():
            bxbk1mask = torch.zeros_like(i2t_labels) # this is the normalcy mask
            ilatextmask = torch.ones_like(i2t_labels) # this is the text padding mask
            if textmask is not None:
                for i in range(len(textmask)):
                    ilatextmask[i,i:i+self.num_cap_per_img] = textmask[i]
                    bxbk1mask[i,i:i+self.num_cap_per_img] = bxbmask[i,i]
                    if i > 0:
                        bxbk1mask[i,0:i] = bxbmask[i,0:i]
                    if i < len(textmask)-1:
                        bxbk1mask[i,i+self.num_cap_per_img:] = bxbmask[i,i+1:]

            if (not negative_only) and self.comp_upfac > 1: # Perform UWP
            
                upscales = torch.ones_like(ilatextmask)
                moved_logits = i2t_logits - ilatextmask * 10
                moved_logits -= i2t_labels * 10
                min_loc = torch.argmin(moved_logits,dim=1)
                row_inds = torch.arange(len(min_loc)).to(min_loc.device)
                upscales[row_inds,min_loc] *= self.comp_upfac
                ilatextmask = ilatextmask * upscales

        ila_loss = -(F.logsigmoid(i2t_labels * i2t_logits) * ilatextmask * (1-bxbk1mask)).sum() / text_features.shape[0] # text-conditioned sigmoid loss

        # KTA loss
        if self.key_token_alignment_loss is not None: # reinforce the loss with additional cross attention on just key tokens
            with torch.no_grad(): # determine key tokens without needing grad
                _, attn_weights = visual_proj(text_features, image_tokens, image_tokens, mask=imgmask,output_attn_weights=True)
                tempattnmap = attn_weights[:,:,:-1] # B,B+K-1,196
                B,H,W = tempattnmap.shape
                k = max(1, int(W * self.key_token_thresh))
                _, topk_indices = torch.topk(tempattnmap, k=k, dim=-1)
                batch_idx = torch.arange(B).view(B, 1, 1).expand(B, H, k)
                row_idx = torch.arange(H).view(1, H, 1).expand(B, H, k)
                newimgmask = torch.zeros_like(tempattnmap,dtype=torch.int)
                newimgmask[batch_idx, row_idx, topk_indices] = 1
                newimgmask = newimgmask.unsqueeze(1).expand(-1,num_heads,-1,-1).reshape(B*num_heads,H,W)
                del attn_weights,batch_idx,row_idx,topk_indices
            # calculate KTA loss
            new_local_image_features = checkpoint(visual_proj,text_features, image_tokens, image_tokens, mask=newimgmask)
            new_local_image_features = F.normalize(new_local_image_features, dim=-1)
            new_i2t_logits = get_ila_logits(new_local_image_features, global_text_features, logit_scale, logit_bias)
            kta_loss = -(F.logsigmoid(i2t_labels * new_i2t_logits) * ilatextmask * (1-bxbk1mask)).sum() / text_features.shape[0]
            all_losses['kta_loss'] = kta_loss
            
        # MPS loss
        if self.added_mps_loss:
            
            g_image_features = F.normalize(image_features, dim=-1)  #(B, D)
            g_text_features = F.normalize(g_text_features, dim=-1)  #(B*K, D)
            mps_logits = get_mps_logits(image_features=g_image_features, text_features=g_text_features,
                                                logit_scale=logit_scale, logit_bias=logit_bias)
            g2g_labels, gtmask = get_mps_ground_truth(device=g_text_features.device,
                                            dtype=g_text_features.dtype,
                                            target_shape=mps_logits.size(),
                                            negative_only=negative_only,
                                            num_captions=self.num_cap_per_img, textmask = textmask)
            bxbkmask = bxbmask.unsqueeze(2).expand(-1,-1,self.num_cap_per_img).reshape(len(bxbmask),-1)
            mps_loss = -(F.logsigmoid(g2g_labels * mps_logits)*gtmask * (1-bxbkmask)).sum() / g_text_features.shape[0] # multi-positive sigmoid loss
                

            loss = self.ila_fac * ila_loss + self.mps_fac*mps_loss
            all_losses['mps_loss'] = mps_loss
        else:
            loss = ila_loss
        all_losses['ila_loss'] = ila_loss
        if 'kta_loss' in all_losses:
            loss += self.key_token_alignment_loss * kta_loss
        # IIS loss, only applies to positive image and text items
        if not negative_only and self.iis_loss is not None:
            row_idx = torch.arange(len(local_image_features)).unsqueeze(1)  # shape (b, 1)
            col_idx = (row_idx + torch.arange(self.num_cap_per_img).unsqueeze(0)).to(local_image_features.device)  # shape (b, k)
            col_idx = col_idx.unsqueeze(-1).expand(-1,-1,global_text_features.size()[-1])
            lif = local_image_features.gather(1,col_idx)
            tif = global_text_features.gather(1,col_idx)
            logits = get_intra_logits(lif,tif, logit_scale=logit_scale, logit_bias=logit_bias) #shape bxkxk
            gts = (torch.eye(self.num_cap_per_img) * 2 - 1).unsqueeze(0).cuda()
            iismask = torch.einsum('bi,bj->bij',textmask,textmask)
            iis_loss = -(F.logsigmoid(gts * logits)*iismask).sum() / g_text_features.shape[0]
            loss += self.iis_loss * iis_loss
            all_losses['iis_loss'] = iis_loss

        if show_all_loss:
            return all_losses

        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, image_tokens=None,
                visual_proj=None, output_dict=False, localloss=False, show_all_loss = False, normals = None, **kwargs):
        """
        image_features: (B, D) global image features
        image_tokens: (B, L, D) local image tokens
        text_features: (B, K. D) text features (K=num_cap_per_img)
        logit_scale: scaling factor for logits
        logit_bias: optional bias for logits
        negative_only: whether all text-image pairs are negative (for distributed training)
        visual_proj: visual projection module
        textmask (from kwargs): since ItemizedCLIP allows varying amount of items per image, this masks which text items are real (1) and which ones are just padding (0)
        localloss: whether to only compute loss over local text/image pairs (for distributed training)
        output_dict: whether to output loss as a dictionary
        show_all_loss: whether to show all loss components in output dictionary
        normals: normalcy labels for the batch
        """

        if self.added_mps_loss:
            g_text_features = text_features  # (B*K, D)
        else:
            g_text_features = None
        
        if len(normals.size()) > 1:
            normals = normals.squeeze(1)
        
        if self.no_normal_check:
            normals = torch.zeros_like(normals)

        # We don't change the shape of image tokens anywhere before the loss function.
        batch_size = image_tokens.shape[0]
        num_captions = self.num_cap_per_img
        textmask = None
        if 'textmask' in kwargs:
            textmask = kwargs['textmask']
        # Sample 1 text item from each image in batch as negative for ILA, KTA and MPS
        caption_indices = torch.arange(batch_size * num_captions).view(batch_size, num_captions).to(
            text_features.device)

        text_features = downsample_text_features(text_features=text_features, batch_size=batch_size,
                                                 caption_indices=caption_indices,
                                                 num_captions=num_captions, textmask = textmask)


        #local loss
        loss = self._loss_with_attn_pool(image_features=image_features,
                                         image_tokens=image_tokens,
                                         text_features=text_features,
                                         visual_proj=visual_proj,
                                         logit_scale=logit_scale,
                                         logit_bias=logit_bias,
                                         g_text_features=g_text_features, textmask = textmask, show_all_loss = show_all_loss, normals1 = normals,normals2=normals)

        # compute loss pairing image from current GPU and text from other GPUs
        if self.world_size > 1 and not localloss:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                normals2_right = normals2_left = normals2 = normals
                if self.added_mps_loss:
                    g_text_features_to_right = g_text_features_to_left = g_text_features
                if textmask is not None:
                    textmask_to_left = textmask_to_right = textmask

                num_bidir, remainder = divmod(self.world_size - 1, 2)

                g_text_features_recv = None  # predefine it to be None

                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    normals2_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        normals2_left,
                        normals2_right,
                    )
                    if textmask is not None:
                        textmask_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            textmask_to_left,
                            textmask_to_right,
                        )
                    if self.added_mps_loss:
                        g_text_features_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            g_text_features_to_left,
                            g_text_features_to_right,
                        )
                        for j in range(len(text_features_recv)):
                            loss = loss_merger(loss, self._loss_with_attn_pool(
                                image_features=image_features,
                                image_tokens=image_tokens,
                                text_features=text_features_recv[j],
                                visual_proj=visual_proj,
                                logit_scale=logit_scale,
                                logit_bias=logit_bias,
                                negative_only=True,
                                g_text_features=g_text_features_recv[j],
                                textmask = textmask_recv[j] if textmask is not None else None,  show_all_loss = show_all_loss,normals1 = normals,normals2 = normals2_recv[j]
                            ))
                    else:
                        for j,f in enumerate(text_features_recv):
                            loss = loss_merger(loss, self._loss_with_attn_pool(
                                image_features=image_features,
                                image_tokens=image_tokens,
                                text_features=f,
                                visual_proj=visual_proj,
                                logit_scale=logit_scale,
                                logit_bias=logit_bias,
                                negative_only=True,
                                g_text_features=None,
                                textmask = textmask_recv[j]  if textmask is not None else None,  show_all_loss = show_all_loss,normals1 = normals,normals2 = normals2_recv[j]))
                    text_features_to_left, text_features_to_right = text_features_recv
                    normals2_left,normals2_right = normals2_recv
                    if self.added_mps_loss:
                        g_text_features_to_left, g_text_features_to_right = g_text_features_recv
                    if textmask is not None:
                        textmask_to_left, textmask_to_right = textmask_recv
                    

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    normals2_recv = neighbour_exchange_with_grad(left_rank,right_rank,normals2_right)
                    if textmask is not None:
                        textmask_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, textmask_to_right)
                    if self.added_mps_loss:
                        g_text_features_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, g_text_features_to_right)
                        loss = loss_merger(loss, self._loss_with_attn_pool(
                            image_features=image_features,
                            image_tokens=image_tokens,
                            text_features=text_features_recv,
                            visual_proj=visual_proj,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            negative_only=True,
                            g_text_features=g_text_features_recv,
                            textmask = textmask_recv  if textmask is not None else None,  show_all_loss = show_all_loss,normals1 = normals,normals2 = normals2_recv)
                        )
                    else:
                        loss = loss_merger(loss, self._loss_with_attn_pool(
                            image_features=image_features,
                            image_tokens=image_tokens,
                            text_features=text_features_recv,
                            visual_proj=visual_proj,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            negative_only=True,
                            g_text_features=None,
                            textmask = textmask_recv  if textmask is not None else None, show_all_loss = show_all_loss,normals1 = normals,normals2 = normals2_recv))
            else:
                text_features_to_right = text_features
                normals2_right = normals2
                if self.added_mps_loss:
                    g_text_features_to_right = g_text_features
                if textmask is not None:
                    textmask_to_right = textmask

                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    normals2_left = neighbour_exchange_with_grad(left_rank,right_rank,normals2_right)

                    if self.added_mps_loss:
                        g_text_features_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, g_text_features_to_right)
                    else:
                        g_text_features_from_left = None

                    
                    if textmask is not None:
                        textmask_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, textmask_to_right)

                    loss = loss_merger(loss, self._loss_with_attn_pool(
                        image_features=image_features,
                        image_tokens=image_tokens,
                        text_features=text_features_from_right,
                        visual_proj=visual_proj,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                        negative_only=True,
                        g_text_features=g_text_features_from_right,
                        textmask = textmask_recv  if textmask is not None else None,  show_all_loss = show_all_loss,normals1 = normals,normals2 = normals2_right))

                    text_features_to_left = text_features_from_right

        return {"contrastive_loss": loss} if (output_dict and not show_all_loss) else loss

# when there are multiple loss components in a dictionary, add each loss into its respective entry in an aggregate dictinary
def loss_merger(loss1,newloss):
    if isinstance(loss1,dict):
        for k in newloss:
            loss1[k] += newloss[k]
    else:
        loss1 += newloss
    return loss1

# Following FLAIR, we pick 1 text item from each image as negative for ILA, KTA and MPS.
def downsample_text_features(text_features, batch_size, caption_indices, num_captions,textmask = None):
    device = text_features.device
    own_caption_indices = caption_indices  # Shape: (B, K)

    mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)

    other_image_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(batch_size, batch_size)
    other_image_indices = other_image_indices[mask].view(batch_size, batch_size - 1)

    if textmask is not None:
        num_avail = textmask.sum(dim=1) # size B
        num_avails = num_avail.unsqueeze(0).expand(batch_size,batch_size) # size BxB
        num_avails = num_avails[mask].view(batch_size, batch_size - 1)
        random_offsets = torch.floor(torch.rand_like(num_avails,dtype=torch.float)*num_avails).long()
    else:
        random_offsets = torch.randint(0, num_captions, (batch_size, batch_size - 1), device=device)  # (B, B-1)
    
    
    other_caption_indices = caption_indices[other_image_indices, random_offsets]  # sampled indices (B, B-1)

    combined_indices = torch.cat([own_caption_indices, other_caption_indices], dim=1)
    combined_indices, _ = combined_indices.sort(dim=1)

    flat_combined_indices = combined_indices.view(-1)  # flatten to take the text_features out

    downsampled_text_features = text_features[flat_combined_indices]

    embed_dim = text_features.shape[-1]  # Reshape to (B, K + B - 1, D)
    downsampled_text_features = downsampled_text_features.view(batch_size, num_captions + batch_size - 1, embed_dim)
    return downsampled_text_features
