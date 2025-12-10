#!/bin/bash -x

cd 2D_images_Tasks/src

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # change based on how many GPUs you have
export SLURM_JOB_NUM_NODES=8                  # Change based on how many GPUs you have

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

torchrun --nnode=1 --nproc_per_node=gpu --rdzv_id=152136 \
    --rdzv_endpoint=localhost:15213 --rdzv_backend=c10d main.py \
    --model ViT-B-16-ItemizedCLIP \
    --use-flair-loss \
    --train-dataset-type webdataset \
    --lr 5e-4 \
    --warmup 2000 \
    --epochs 100  \
    --num-sampled-captions 7 \
    --log-every-n-steps 10 \
    --train-data '../../../flair/datasets/scraped_cc3m_subset/{00000..00057}.tar' \
    --val-data '../../../flair/datasets/scraped_cc3m_subset/{00058..00059}.tar' \
    --train-num-samples 300000 \
    --batch-size 128 \
    --precision amp \
    --workers 7 \
    --beta1 0.9 \
    --beta2 0.98 \
    --wd 0.8 \
    --eps 1e-8 \
    --report-to wandb \
    --external-captions '../first_300000_rewrites.json' \
    --wandb-project-name itemizedcc0.3m-train \
    --caption-sampling-mode diverse_external \
    --caption-pad-to-train 7 \
    --caption-pad-to-val 7 \
    --add-mps-loss \
    --global-retrieval \
    --iis-loss 0.1 \
    --ila-fac 1.0 \
    --mps-fac 0.1 \
    --comp-upfac 1.5 \
    --ila-mask-rate 0.8 \
    --val-flair-retrieval-metrics \
    --key-token-alignment-loss 1.0 \
    --key-token-thresh 0.2 \
    --logs-dir  ../../../flairlogs \



