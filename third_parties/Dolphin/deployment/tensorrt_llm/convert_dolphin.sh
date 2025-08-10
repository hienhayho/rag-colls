#!/usr/bin/env bash
set -ex

############################################################################################
# Reference: https://github.com/NVIDIA/TensorRT-LLM/tree/v0.18.2/examples/multimodal#nougat
############################################################################################

export LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/tensorrt_libs/:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH

# 1. Download Huggingface weights
export MODEL_NAME="Dolphin"
git clone https://huggingface.co/Bytedance/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}


export MAX_BATCH_SIZE=16
export MAX_SEQ_LEN=4096
export MAX_INPUT_LEN=10
export MAX_ENCODER_INPUT_LEN=784

# 2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in examples/enc_dec
python ./convert/convert_checkpoint.py --model_type bart \
    --model_dir tmp/hf_models/${MODEL_NAME} \
    --output_dir tmp/trt_models/${MODEL_NAME}/bfloat16 \
    --tp_size 1 \
    --pp_size 1 \
    --dtype bfloat16 \
    --nougat


trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/bfloat16/decoder \
    --output_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16/decoder \
    --paged_kv_cache disable \
    --moe_plugin disable \
    --gemm_plugin bfloat16 \
    --bert_attention_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --remove_input_padding enable \
    --max_beam_width 1 \
    --max_batch_size ${MAX_BATCH_SIZE} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --max_input_len ${MAX_INPUT_LEN} \
    --max_encoder_input_len $((${MAX_BATCH_SIZE} * ${MAX_ENCODER_INPUT_LEN})) # MAX_BATCH_SIZE (max_batch_size) * MAX_ENCODER_INPUT_LEN (num_visual_features)

# 3. Generate TensorRT engines for visual components and combine everything into final pipeline.
python ./convert/build_visual_engine.py --model_type nougat \
    --model_path tmp/hf_models/${MODEL_NAME} \
    --max_batch_size ${MAX_BATCH_SIZE}
