#!/bin/bash -x

cd 2D_images_Tasks/src

export CUDA_VISIBLE_DEVICES=0


torchrun --nproc_per_node 1 --rdzv_endpoint=localhost:29438 -m main \
    --model ViT-B-16-ItemizedCLIP \
    --resume /path_to_your_checkpoints/checkpoints/epoch_60.pt \
    --classification-dataset rsicd \
    --batch-size 128 \
    --precision amp \
    --workers 8 \
    --dist-url env://localhost:29438 \
    --use-flair-loss \
    --inference-with-flair \
    --rsicd-data-dir ../../../rsicd \
