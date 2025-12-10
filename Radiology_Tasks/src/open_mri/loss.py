"""
More traditional CLIP loss implementations
"""

import sys
sys.path.append('.')
sys.path.append('..')

from open_clip.loss import *


class CustomClipLoss(ClipLoss):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        local_global_text = False
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.local_global_text = local_global_text

    @staticmethod
    def _gather_false_negatives(
            false_negatives,
            world_size=1,
            use_horovod=False
    ):
        assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
        if use_horovod:
            raise NotImplementedError
        else:
            # We gather tensors from all gpus
            gathered_false_negatives = [torch.zeros_like(false_negatives) for _ in range(world_size)]
            dist.all_gather(gathered_false_negatives, false_negatives)
            all_false_negatives = torch.cat(gathered_false_negatives, dim=0)

        return all_false_negatives
    
    def get_false_negatives(self, false_negatives):
        if self.world_size > 1:
            all_false_negatives = self._gather_false_negatives(false_negatives, world_size=self.world_size, use_horovod=self.use_horovod)
            if self.local_loss:
                false_negatives = false_negatives @ all_false_negatives.T
            else:
                false_negatives = all_false_negatives @ all_false_negatives.T
        else:
            false_negatives = false_negatives @ false_negatives.T

        return false_negatives

    def get_ground_truth(self, device, num_logits):
        """
        Returns a binary matrix for the alternative loss formulation.
        
        If world_size == 1 or local_loss is False:
            - Returns an identity matrix of shape [num_logits, num_logits].
        
        If world_size > 1 and local_loss is True:
            - The logits are expected to have shape [num_logits, num_logits * world_size],
            so we construct a target matrix of that shape. Only the columns corresponding
            to the current rank (i.e. columns [rank*num_logits:(rank+1)*num_logits]) form an
            identity matrix (indicating the positives); all other positions are zeros.
        """
        if self.prev_num_logits != num_logits or device not in self.labels:
            if self.world_size > 1 and self.local_loss:
                labels = torch.zeros((num_logits, num_logits * self.world_size), device=device)
                start = self.rank * num_logits
                end = (self.rank + 1) * num_logits
                labels[:, start:end] = torch.eye(num_logits, device=device)
            else:
                labels = torch.eye(num_logits, device=device)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels
    
    def forward(self, image_features, text_features, logit_scale, false_negatives, output_dict=False):
        """
        Instead of using F.cross_entropy with index targets, we use the log_softmax
        multiplied elementwise by a binary target matrix.
        """
        if self.local_global_text:
            text_features, _ = text_features
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        # Get one-hot targets (identity matrix) for the batch
        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        if false_negatives is not None:
            # Get false negatives (binary matrix) for the batch
            false_negatives = self.get_false_negatives(false_negatives)
            labels = torch.max(labels, false_negatives)

            # If you do normalization like this:
            # labels = labels / labels.sum(dim=1, keepdim=True).
            # The result will not change no matter you use false-negative or not.

        # Compute the loss by summing the log probabilities at the positive positions.
        # For a one-hot target, this is equivalent to the standard cross entropy.
        loss_i2t = - torch.sum(F.log_softmax(logits_per_image, dim=1) * labels, dim=1).mean()
        loss_t2i = - torch.sum(F.log_softmax(logits_per_text, dim=1) * labels, dim=1).mean()
        total_loss = (loss_i2t + loss_t2i) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

    

class CustomSigLipLoss(CustomClipLoss):
    """ SigLIP but without the gathering optimization.
    """
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        local_global_text = False
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        
        self.local_global_text = local_global_text
    
    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits = logit_scale * image_features @ all_text_features.T
            else:
                logits = logit_scale * all_image_features @ all_text_features.T
        else:
            logits = logit_scale * image_features @ text_features.T

        if logit_bias is not None:
            logits += logit_bias

        return logits
    
    def get_ground_truth(self, device, num_logits):
        if self.prev_num_logits != num_logits or device not in self.labels:
            if self.world_size > 1 and self.local_loss:
                labels = -torch.ones((num_logits, num_logits * self.world_size), device=device)
                start = self.rank * num_logits
                end = (self.rank + 1) * num_logits
                labels[:, start:end] += 2 * torch.eye(num_logits, device=device)
            else:
                labels = -torch.ones((num_logits, num_logits), device=device)
                labels += 2 * torch.eye(num_logits, device=device)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels
    
    def forward(self, image_features, text_features, logit_scale, logit_bias=None, false_negatives=None, output_dict=False):
        if self.local_global_text:
            text_features, _ = text_features
        device = image_features.device
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)

        # Get sigmoid targets (1 & -1) for the batch
        labels = self.get_ground_truth(device, logits.shape[0])

        if false_negatives is not None:
            # Get false negatives (binary matrix) for the batch
            false_negatives = self.get_false_negatives(false_negatives)
            false_negatives.fill_diagonal_(0)
            false_negatives = torch.ones_like(false_negatives) - false_negatives
            loss = -F.logsigmoid(labels * logits)[false_negatives.bool()].sum() / logits.shape[0]
        else:
            loss = -F.logsigmoid(labels * logits).sum() / logits.shape[0]

        return {"contrastive_loss": loss} if output_dict else loss
