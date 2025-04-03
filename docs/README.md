# rag-colls's Documentation

> Refer to [vllm's docs](https://github.com/vllm-project/vllm/blob/main/docs/README.md).

## Installation

- Make sure in `docs` directory.

```bash
cd docs/
```

- Install `make`:

```bash
# Linux
apt-get update && apt-get install make -y

#macOS
brew install make
```

- Install the dependencies:

```bash
deactivate

uv venv ../docs_env

source ../docs_env/bin/activate

uv pip install -r requirements.txt
```

## Build the docs

- Clean the previous build (optional but recommended):

```bash
make clean
```

- Generate the HTML Document:

```bash
make html
```

## Open the docs with your browser

Serve the documentation locally:

```bash
uv run --active python -m http.server -d build/html
```

It will run on port `8000` by default. You can run with another port, for example:

```bash
uv run --active python -m http.server 3000 -d build/html
```
