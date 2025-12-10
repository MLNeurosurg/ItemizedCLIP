#!/bin/bash

cd Radiology_Tasks/src/open_mri_train

torchrun --rdzv_endpoint=localhost:29423 --nproc_per_node 8 main.py \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --report-to wandb \
        --train-data train \
        --val-data valid \
        --warmup 2000 \
        --batch-size 8 \
        --accum-batch 4 \
        --lr 1.75e-4 \
        --wd 0.2 \
        --epochs 24 \
        --precision amp \
        --workers 8 \
        --grad-checkpointing \
        --model HLIPSN_BiomedBERT_81616_itemizedclip \
        --dist-url env://localhost:29423 \
        --use-serienames \
        --use-itemizedclip-loss \
        --add-mps-loss \
        --iis-loss 1.0 \
        --mps-fac 0.01 \
	--ila-fac 1.0 \
	--ila-mask-rate 0.9 \
        --num-sampled-captions 7 \
        --comp-upfac 1.5 \
	--wandb-project-name open-ct-itemizedclip \
	--key-token-alignment-loss 1.0 \
        --logs ../logs \
        --ct
