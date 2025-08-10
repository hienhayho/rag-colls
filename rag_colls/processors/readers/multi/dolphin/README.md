# Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting

Integrated from [Dolphin](https://github.com/bytedance/Dolphin). Example usage can be found here: [Colab](https://colab.research.google.com/drive/1f4aGzFvUKFLwXOo2X6WZ3oiNeANZISJK)

## Usage

**1**. Installation

```bash
pip install rag-colls rag-colls[dolphin]
```

**2**. Quickstart:

`DolphinReader` can be used to read `pdf`, `jpg` and `png` files.

```python
from loguru import logger
from pathlib import Path

from rag_colls.processors.readers.multi.dolphin import DolphinReader

reader = DolphinReader(gpu_memory_utilization=0.9)

file_path = Path("samples/data/2503.20376v1.pdf")

documents = reader.load_data(
    file_path=file_path, should_split=True, extra_info={}, encoding="utf-8"
)

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
```
