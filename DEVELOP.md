# Developer guidance

## Installation

Please follow these instruction to install rag-colls in dev mode.

- **Clone the repository**

```bash
git clone https://github.com/hienhayho/rag-colls.git
cd rag-colls/
```

- **Install uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **Install packages:**

```bash
uv pip install -v -e .
uv pip install pytest
```

- **Install pre-commit:**

```bash
uv pip install pre-commit

pre-commit install
```

## Building

Please refer to latest document for developing new features.
