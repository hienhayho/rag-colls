try:
    from agentic_doc.parse import parse
except ImportError:
    raise ImportError(
        "The 'agentic-doc' package is required for this module. "
        "Please install it using 'pip install agentic-doc'."
    )

from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader

load_dotenv()


class AgenticDocReader(BaseReader):
    """
    Reader using the agentic-doc library to parse documents.
    """

    def __init__(self) -> None:
        logger.info("AgenticDocReader initialized !!!")

    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
    ) -> list[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        file_name = file_path.name

        if not extra_info:
            extra_info = {}

        extra_info["file_name"] = file_name
        extra_info["file_path"] = str(file_path)
        extra_info["should_split"] = (
            False  # Disable internal splitting since agentic-doc handles it
        )

        results = parse(documents=[str(file_path)])[0]
        chunks = [
            Document(
                document=doc.text,
                metadata={
                    **extra_info,
                    "source": str(file_path),
                    "chunk_type": doc.chunk_type.name,
                },
            )
            for doc in results.chunks
        ]

        return chunks
