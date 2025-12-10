#!/bin/bash -x

cd 2D_images_Tasks/src

export CUDA_VISIBLE_DEVICES=0

python3 visualize/main.py \
    --model ViT-B-16-ItemizedCLIP \
    --use-flair-loss \
    --resume /scratch/tocho_root/tocho99/yiweilyu/flairlogs/2025_10_18-16_44_03-model_ViT-B-16-FLAIR-lr_0.0003-b_128-j_8-p_amp/checkpoints/epoch_60.pt \
    --train-dataset-type rsicd  \
    --val-data 'rsicd' \
    --batch-size 1 \
    --precision amp \
    --workers 7 \
    --external-captions '/nfs/turbo/umms-tocho/code/yiwei/flair/preprocess/trainv1.json;/nfs/turbo/umms-tocho/code/yiwei/flair/preprocess/testv1.json' \
    --caption-sampling-mode external \
    --rsicd-data-dir ../../../rsicd \
    --viz-id 323



