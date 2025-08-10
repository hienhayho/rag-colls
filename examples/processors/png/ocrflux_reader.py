from pathlib import Path

from loguru import logger

from rag_colls.processors.readers.multi.ocrflux import OCRFluxReader

reader = OCRFluxReader(
    dtype="half",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    max_model_len=6000,
    download_dir="./model_cache",
)

file_path = Path("samples/data/image.png")

documents = reader.load_data(file_path=file_path, should_split=True, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
