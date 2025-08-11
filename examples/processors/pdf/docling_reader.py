from loguru import logger
from pathlib import Path

from rag_colls.processors.readers.multi.docling import DoclingReader, ExportFormat

reader = DoclingReader(export_format=ExportFormat.MARKDOWN)

file_path = Path("samples/data/2503.20376v1.pdf")

documents = reader.load_data(file_path=file_path, should_split=True, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
