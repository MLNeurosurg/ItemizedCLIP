#!/bin/bash -x

cd 2D_images_Tasks/src

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SLURM_JOB_NUM_NODES=8

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

torchrun --nnode=1 --nproc_per_node=gpu --rdzv_id=152136 \
    --rdzv_endpoint=localhost:15213 --rdzv_backend=c10d main.py \
    --model ViT-B-16-ItemizedCLIP \
    --use-flair-loss \
    --train-dataset-type rsicd  \
    --lr 3e-4 \
    --warmup 100 \
    --epochs 100  \
    --save-frequency 5 \
    --num-sampled-captions 6 \
    --log-every-n-steps 100 \
    --train-data 'rsicd' \
    --val-data 'rsicd' \
    --train-num-samples 10240 \
    --dataset-type rsicd \
    --batch-size 128 \
    --precision amp \
    --workers 8 \
    --beta1 0.9 \
    --beta2 0.98 \
    --wd 1.5 \
    --eps 1e-8 \
    --report-to wandb \
    --wandb-project-name itemized_rsicd \
    --caption-sampling-mode diverse_external \
    --caption-pad-to-train 6 \
    --caption-pad-to-val 6 \
    --add-mps-loss \
    --iis-loss 1.0 \
    --mps-fac 2.0 \
    --comp-upfac 1.5 \
    --ila-mask-rate 0.6 \
    --val-flair-retrieval-metrics \
    --key-token-alignment-loss 1.5 \
    --key-token-thresh 0.2 \
    --logs-dir  ../../../flairlogs \
    --rsicd-data-dir ../../../rsicd \
    --external-captions '../../../flair/preprocess/trainv1.json;../../../flair/preprocess/testv1.json' \



