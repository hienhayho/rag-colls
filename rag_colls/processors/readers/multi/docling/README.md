# Docling

This is a simple wrapper of [docling](https://github.com/docling-project/docling) library.

## Usage

**1**. Installation

```bash
pip install rag-colls docling
```

**2**. Quickstart:

`DoclingReader` can handle multiple file formats — see the [supported formats list](https://docling-project.github.io/docling/usage/supported_formats/) for details.

You can optionally provide the `document_converter` parameter, which you can configure as explained in the [usage guide](https://docling-project.github.io/docling/usage/). You can also specify the output format, which can be `html`, `json`, or `markdown`.

```python
from loguru import logger
from pathlib import Path

from rag_colls.processors.readers.multi.docling import DoclingReader, ExportFormat

reader = DoclingReader(export_format=ExportFormat.MARKDOWN)

file_path = Path("samples/data/2503.20376v1.pdf")

documents = reader.load_data(file_path=file_path, should_split=True, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
```
