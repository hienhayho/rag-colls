#!/usr/bin/env bash
set -ex

export MODEL_NAME="Dolphin"

python run_dolphin.py \
    --batch_size 1 \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir tmp/trt_engines/${MODEL_NAME}/vision_encoder \
    --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16 \
    --max_new_tokens 4096 \
    --repetition_penalty 1.0 \
    --input_text "Parse the reading order of this document." \
    --image_path "../../demo/page_imgs/page_1.jpeg"


python run_dolphin.py \
    --batch_size 1 \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir tmp/trt_engines/${MODEL_NAME}/vision_encoder \
    --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16 \
    --max_new_tokens 4096 \
    --repetition_penalty 1.0 \
    --input_text "Read text in the image." \
    --image_path "../../demo/element_imgs/block_formula.jpeg"


python run_dolphin.py \
    --batch_size 1 \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir tmp/trt_engines/${MODEL_NAME}/vision_encoder \
    --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16 \
    --max_new_tokens 4096 \
    --repetition_penalty 1.0 \
    --input_text "Read text in the image." \
    --image_path "../../demo/element_imgs/para_1.jpg"


python run_dolphin.py \
    --batch_size 1 \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir tmp/trt_engines/${MODEL_NAME}/vision_encoder \
    --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16 \
    --max_new_tokens 4096 \
    --repetition_penalty 1.0 \
    --input_text "Parse the table in the image." \
    --image_path "../../demo/element_imgs/table_1.jpeg"
