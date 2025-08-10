# OCRFlux

This is the simple wrapper of [`OCRFlux`](https://github.com/chatdoc-com/OCRFlux).

## Usage

**1**. Installation

```bash
sudo apt-get update
sudo apt-get install poppler-utils poppler-data ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

pip install -U rag-colls rag-colls[ocrflux_py]
```

**2**. Quickstart:

`OCRFluxReader` can be used to read `pdf`, `jpg` and `png` files.

```python
from pathlib import Path

from loguru import logger

from rag_colls.processors.readers.multi.ocrflux import OCRFluxReader

reader = OCRFluxReader(
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    max_model_len=6000,
    download_dir="./model_cache",
)

file_path = Path("samples/data/2503.20376v1.pdf")

documents = reader.load_data(file_path=file_path, should_split=True, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
```
