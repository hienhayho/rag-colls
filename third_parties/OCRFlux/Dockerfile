FROM ubuntu:24.04

WORKDIR /OCRFlux

ENV LANG=en_US.UTF-8 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=true \
    PIP_NO_CACHE_DIR=true \
    PIP_DISABLE_PIP_VERSION_CHECK=true \
    PYTHONPATH=/OCRFlux

SHELL ["/bin/bash", "-c"]

RUN --mount=type=bind,source=./,target=/builder \
    cp -a /builder/. /OCRFlux/ && \
    set -o pipefail && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
        curl \
        fonts-crosextra-caladea \
        fonts-crosextra-carlito \
        gsfonts \
        lcdf-typetools \
        locales \
        msttcorefonts \
        poppler-utils \
        poppler-data \
        python3.12-dev \
        python3.12-full \
        software-properties-common \
        ttf-mscorefonts-installer && \
    locale-gen en_US.UTF-8 && \
    curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.12 /tmp/get-pip.py && \
    python3.12 -m pip install . --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/ && \
    rm -rf ./* \
        /var/lib/apt/lists/* \
        /tmp/* \
        /root/.cache/pip &&\
    find /var/log /var/cache -type f -delete

ENTRYPOINT ["python3.12", "-m", "ocrflux.pipeline"]
