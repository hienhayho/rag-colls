# MarkItDown

This is a simple wrapper of [markitdown](https://github.com/microsoft/markitdown) library.

## Usage

**1**. Installation

```bash
pip install rag-colls markitdown[all]
```

**2**. Quickstart:

`MarkItDownReader` supports reading multiple file formats — see the [documentation](https://github.com/microsoft/markitdown?tab=readme-ov-file#markitdown) for details.

You can optionally provide the `markitdown_converter` parameter, which you can configure as described in the [documentation](https://github.com/microsoft/markitdown?tab=readme-ov-file#python-api).

```python
from loguru import logger
from pathlib import Path

from rag_colls.processors.readers.multi.markitdown import MarkItDownReader

reader = MarkItDownReader()

file_path = Path("samples/data/2503.20376v1.pdf")

documents = reader.load_data(file_path=file_path, should_split=True, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
```
