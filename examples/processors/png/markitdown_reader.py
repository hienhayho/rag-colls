from loguru import logger
from pathlib import Path

from openai import OpenAI
from markitdown import MarkItDown

from rag_colls.processors.readers.multi.markitdown import MarkItDownReader

reader = MarkItDownReader(
    markitdown_converter=MarkItDown(llm_client=OpenAI(), llm_model="gpt-4o-mini")
)

file_path = Path("samples/data/image.png")

documents = reader.load_data(file_path=file_path, should_split=True, extra_info={})

logger.info(f"Loaded {len(documents)} documents from {file_path}")

print(f"First document content: {documents[0].document}")
