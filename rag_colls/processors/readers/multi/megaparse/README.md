# MegaParse

This is a simple wrapper of [megaparse](https://github.com/QuivrHQ/MegaParse) library.

## Usage

**1**. Installation

```bash
pip install rag-colls megaparse
pip install "unstructured[all-docs]==0.17.2"
```

**2**. Quickstart:

`MegaParseReader` can handle `pdf`, `pptx` and `docx`. See more in [documentation](https://github.com/QuivrHQ/MegaParse?tab=readme-ov-file#support).

- `Default` version:

```python
from loguru import logger
from pathlib import Path

from rag_colls.processors.readers.multi.megaparse import MegaParseReader

reader = MegaParseReader()

file_path = Path("samples/data/2503.20376v1.pdf")

documents = reader.load_data(file_path=file_path, should_split=True, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
```

- `Vision` version:

```python
from loguru import logger
from pathlib import Path

from rag_colls.processors.readers.multi.megaparse import MegaParseReader
from megaparse.parser.megaparse_vision import MegaParseVision
from langchain_openai import ChatOpenAI

reader = MegaParseReader(
    megaparse_converter=MegaParseVision(model=ChatOpenAI(model="gpt-4o-mini"))
)

file_path = Path("samples/data/2503.20376v1.pdf")

documents = reader.load_data(file_path=file_path, should_split=True, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
```
