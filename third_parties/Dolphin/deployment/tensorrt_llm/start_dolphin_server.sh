#!/usr/bin/env bash
set -ex

export MODEL_NAME="Dolphin"

python api_server.py \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir tmp/trt_engines/${MODEL_NAME}/vision_encoder \
    --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16 \
    --max_batch_size 16
