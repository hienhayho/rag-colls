from pathlib import Path

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader


class DocxReader(BaseReader):
    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
        encoding: str = "utf-8",
    ) -> list[Document]:
        try:
            import docx
        except ImportError:
            raise ImportError(
                "python-docx is not installed. Please install it with 'pip install python-docx'."
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".docx":
            raise ValueError(f"File must be a DOCX file: {file_path}")

        # Read the DOCX file
        doc = docx.Document(file_path)

        if not extra_info:
            extra_info = {}

        extra_info["file_path"] = str(file_path)
        extra_info["should_split"] = should_split
        extra_info["file_size"] = file_path.stat().st_size
        extra_info["file_type"] = "docx"
        extra_info["source"] = file_path.name

        # TODO: This currently only supports to split each paragraph into a document
        text = []
        for _, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():  # Only add non-empty paragraphs
                text.append(paragraph.text)

        documents = [
            Document(document="\n".join(text).encode(encoding), metadata=extra_info)
        ]

        return documents
