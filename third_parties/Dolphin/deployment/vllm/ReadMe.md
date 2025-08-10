<h1 align="center">
🚀 Dolphin vLLM Demo
</h1>

## ✅ Introduction
The Dolphin model employs a **Swin Encoder + MBart Decoder** architecture. In the HuggingFace Transformers [Config](https://huggingface.co/ByteDance/Dolphin/blob/main/config.json),
its architectures field is specified as "VisionEncoderDecoderModel". vLLM does not natively support this architecture.
To enable vLLM deployment of the Dolphin model, we implemented two vllm plugins: [vllm-dolphin](https://github.com/hanyd2010/vllm-dolphin)[![PyPI version](https://img.shields.io/pypi/v/vllm-dolphin)](https://pypi.org/project/vllm-dolphin/) and [vllm-mbart](https://github.com/hanyd2010/vllm-mbart)[![PyPI version](https://img.shields.io/pypi/v/vllm-mbart)](https://pypi.org/project/vllm-mbart/).
We also provide Dolphin vllm demos for both offline inference and online deployment.

## 🛠️ Installation

```
# Install vllm
pip install vllm>=0.9.0

# Install vllm-dolphin
pip install vllm-dolphin==0.1
```

## ⚡ Offline Inference
```
# predict elements reading order
python deployment/vllm/demo_vllm.py --model ByteDance/Dolphin --image_path ./demo/page_imgs/page_1.jpeg --prompt "Parse the reading order of this document."

# recognize text/latex
python deployment/vllm/demo_vllm.py --model ByteDance/Dolphin --image_path ./demo/element_imgs/block_formula.jpeg --prompt "Read text in the image."
python deployment/vllm/demo_vllm.py --model ByteDance/Dolphin --image_path ./demo/element_imgs/para_1.jpg --prompt "Read text in the image."

# recognize table
python deployment/vllm/demo_vllm.py --model ByteDance/Dolphin --image_path ./demo/element_imgs/table_1.jpeg --prompt "Parse the table in the image."
```


## ⚡ Online Inference
```
# 1. Start Api Server
python deployment/vllm/api_server.py --model="ByteDance/Dolphin" --hf-overrides "{\"architectures\": [\"DolphinForConditionalGeneration\"]}"

# 2. Predict
# predict elements reading order
python deployment/vllm/api_client.py --image_path ./demo/page_imgs/page_1.jpeg --prompt "Parse the reading order of this document."

# recognize text/latex
python deployment/vllm/api_client.py --image_path ./demo/element_imgs/block_formula.jpeg --prompt "Read text in the image."
python deployment/vllm/api_client.py --image_path ./demo/element_imgs/para_1.jpg --prompt "Read text in the image."

# recognize table
python deployment/vllm/api_client.py --image_path ./demo/element_imgs/table_1.jpeg --prompt "Parse the table in the image."
```
