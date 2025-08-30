# Agentic-doc

This is a simple wrapper of [agentic-doc](https://github.com/landing-ai/agentic-doc) library. Please check [documentation](https://github.com/landing-ai/agentic-doc?tab=readme-ov-file#quick-start) for API key.

## Usage

**1**. Installation

```bash
pip install rag-colls agentic-doc
```

**2**. Quickstart:

`AgenticDocReader` can handle `pdf`, `jpg` and `png`. See more in [documentation](https://landing.ai/agentic-document-extraction).

```python
from loguru import logger
from pathlib import Path

from rag_colls.processors.readers.multi.agentic_doc import AgenticDocReader

reader = AgenticDocReader()

file_path = Path("samples/data/2503.20376v1.pdf")

documents = reader.load_data(file_path=file_path, should_split=False, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
```
