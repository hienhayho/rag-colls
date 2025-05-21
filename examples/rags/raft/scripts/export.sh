#!/bin/bash
# output/vx-xxx/checkpoint-xxx \

CHECKPOINT=$1
OUT=$2

swift export \
    --adapters "$CHECKPOINT" \
    --output_dir "$OUT" \
    --merge_lora true \
    --use_hf true
