#!/bin/bash
# Qwen/Qwen2.5-3B-Instruct

MODEL=$1
DATA=$2
OUT=$3
E=${4:-"3"}

NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \
    swift sft \
    --model "$MODEL" \
    --train_type lora \
    --dataset "$DATA" \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --num_train_epochs "$E" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --deepspeed zero2 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 50 \
    --max_length 8196 \
    --output_dir "$OUT" \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 2 \
    --use_hf true
