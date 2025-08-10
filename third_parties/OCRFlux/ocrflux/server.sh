#!/bin/bash

model_path=$1
port=$2
vllm serve $model_path --port $port --max-model-len 8192 --gpu_memory_utilization 0.8
