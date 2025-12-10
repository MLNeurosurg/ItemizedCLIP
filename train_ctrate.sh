#!/bin/bash

cd Radiology_Tasks/src/open_mri_train

torchrun --rdzv_endpoint=localhost:29427 --nproc_per_node 8 main.py \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --report-to wandb \
        --train-data itemized \
        --zeroshot-ct-rate \
        --warmup 100 \
        --batch-size 16 \
        --accum-batch 4 \
        --lr 1e-4 \
        --wd 0.5 \
        --epochs 80 \
        --precision amp \
        --workers 8 \
        --grad-checkpointing \
        --model CTRATE_HLIP_BiomedBERT_itemizedclip \
        --dist-url env://localhost:29427 \
        --use-itemizedclip-loss \
        --add-mps-loss \
        --iis-loss 1.0 \
        --mps-fac 0.1 \
	--ila-fac 1.0 \
	--ila-mask-rate 0.95 \
        --num-sampled-captions 10 \
        --comp-upfac 1.0 \
	--wandb-project-name itemizedclip-ctrate \
	--key-token-alignment-loss 1.0 \
        --is-ct-rate \
        --zeroshot-ct-rate \
        --save-and-eval-every 1 \
        --log-every-n-steps 10 \
        --use-cxr-bert \
        --logs ../logs \
        --lock-text