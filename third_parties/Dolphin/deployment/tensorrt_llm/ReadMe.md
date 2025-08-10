<h1 align="center">
🚀 Dolphin TensorRT-LLM Demo
</h1>

## ✅ Introduction
The Dolphin model employs a **Swin Encoder + MBart Decoder** architecture. In the HuggingFace Transformers [Config](https://huggingface.co/ByteDance/Dolphin/blob/main/config.json),
its architectures field is specified as "VisionEncoderDecoderModel". **Dolphin**, **[Nougat](https://huggingface.co/docs/transformers/model_doc/nougat)**, and **[Donut](https://huggingface.co/docs/transformers/model_doc/donut)** share the same model architecture. TensorRT-LLM has already supported the Nougat model.
Following Nougat's conversion script, we have successfully implemented Dolphin on TensorRT-LLM.

**Note:** [prompt_ids](./dolphin_runner.py#L120) MUST be of **int32** type, otherwise TensorRT-LLM will produce incorrect results.

## 🛠️ Installation
> We only test TensorRT-LLM 0.18.1 on Linux.

https://nvidia.github.io/TensorRT-LLM/0.18.1/installation/linux.html


## ⚡ Offline Inference
```
export MODEL_NAME="Dolphin"

# predict elements reading order
python run_dolphin.py \
    --batch_size 1 \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir tmp/trt_engines/${MODEL_NAME}/vision_encoder \
    --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16 \
    --max_new_tokens 4096 \
    --repetition_penalty 1.0 \
    --input_text "Parse the reading order of this document." \
    --image_path "../../demo/page_imgs/page_1.jpeg"

# recognize text/latex
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

# recognize table
python run_dolphin.py \
    --batch_size 1 \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir tmp/trt_engines/${MODEL_NAME}/vision_encoder \
    --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16 \
    --max_new_tokens 4096 \
    --repetition_penalty 1.0 \
    --input_text "Parse the table in the image." \
    --image_path "../../demo/element_imgs/table_1.jpeg"
```


## ⚡ Online Inference
```
# 1. Start Api Server
export MODEL_NAME="Dolphin"

python api_server.py \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir tmp/trt_engines/${MODEL_NAME}/vision_encoder \
    --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16 \
    --max_batch_size 16

# 2. Predict
# predict elements reading order
python deployment/tensorrt_llm/api_client.py --image_path ./demo/page_imgs/page_1.jpeg --prompt "Parse the reading order of this document."

# recognize text/latex
python deployment/tensorrt_llm/api_client.py --image_path ./demo/element_imgs/block_formula.jpeg --prompt "Read text in the image."
python deployment/tensorrt_llm/api_client.py --image_path ./demo/element_imgs/para_1.jpg --prompt "Read text in the image."

# recognize table
python deployment/tensorrt_llm/api_client.py --image_path ./demo/element_imgs/table_1.jpeg --prompt "Parse the table in the image."
```
