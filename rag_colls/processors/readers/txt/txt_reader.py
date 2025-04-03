from pathlib import Path

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader


class TxtReader(BaseReader):
    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
        encoding: str = "utf-8",
    ) -> list[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the text file
        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()
        file_name = file_path.name

        if not extra_info:
            extra_info = {}

        extra_info["file_path"] = str(file_path)
        extra_info["file_size"] = file_path.stat().st_size
        extra_info["file_type"] = "txt"
        extra_info["encoding"] = encoding
        extra_info["should_split"] = should_split

        # Return the entire text as one document
        return [
            Document(
                document=text,
                metadata=dict(
                    source=file_name,
                    **extra_info,
                ),
            )
        ]
