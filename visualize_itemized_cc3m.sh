#!/bin/bash -x

cd 2D_images_Tasks/src

export CUDA_VISIBLE_DEVICES=0

python3 visualize/main.py \
    --model ViT-B-16-ItemizedCLIP \
    --use-flair-loss \
    --resume /checkpoints/epoch_X.pt \
    --train-dataset-type webdataset  \
    --num-sampled-captions 7 \
    --val-data '../../../flair/datasets/scraped_cc3m_subset/{00058..00059}.tar' \
    --batch-size 1 \
    --precision amp \
    --workers 7 \
    --external-captions '../first_300000_rewrites.json' \
    --caption-sampling-mode external \
    --viz-id 80



