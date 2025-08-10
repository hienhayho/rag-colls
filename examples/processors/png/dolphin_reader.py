from loguru import logger
from pathlib import Path

from rag_colls.processors.readers.multi.dolphin import DolphinReader

reader = DolphinReader(gpu_memory_utilization=0.5)

file_path = Path("samples/data/image.png")

documents = reader.load_data(file_path=file_path, should_split=True, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
