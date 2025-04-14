FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/hienhayho/scripts/refs/heads/main/scripts/intall_fzf.sh | bash
