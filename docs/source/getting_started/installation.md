# Installation

You can follow one of these methods:

## Install from PyPI

The easiest way to install the latest stable version of `rag-colls` is via `pip`:

```bash
pip install -U rag-colls
```

## Install from source

If you want to install the latest development version, you can clone the repository and install it manually:

```bash
# Clone the repository
git clone https://github.com/hienhayho/rag-colls.git
cd rag-colls/

pip install -v .
```

## Install with docker (CPU)

If you prefer to use Docker, you can build the image from the Dockerfile provided in the repository:

```bash
# Clone the repository
git clone https://github.com/hienhayho/rag-colls.git
cd rag-colls/

# Choose python version and setup OPENAI_API_KEY
export PYTHON_VERSION="3.10"
export OPENAI_API_KEY="your-openai-api-key-here"

# Docker build
DOCKER_BUILDKIT=1 docker build \
                -f docker/Dockerfile \
                --build-arg OPENAI_API_KEY="$OPENAI_API_KEY" \
                --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
                -t rag-colls:$PYTHON_VERSION .

docker run -it --name rag_colls --shm-size=2G rag-colls:$PYTHON_VERSION
```
