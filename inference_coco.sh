#!/bin/bash -x

cd 2D_images_Tasks/src

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node 1 --rdzv_endpoint=localhost:29435 -m main \
    --model ViT-B-16-ItemizedCLIP \
    --resume /checkpoints/epoch_56.pt \
    --coco-data-root-dir  ../../../flair/datasets/coco \
    --retrieval-coco \
    --batch-size 128 \
    --precision amp \
    --workers 8 \
    --dist-url env://localhost:29435 \
    --use-flair-loss \
    --inference-with-flair
    


    



