from pathlib import Path

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader


class PyMuPDFReader(BaseReader):
    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
        encoding: str = "utf-8",
    ) -> list[Document]:
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF is not installed. Please install it with 'pip install PyMuPDF'."
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        docs = fitz.open(file_path)
        file_name = file_path.name

        if not extra_info:
            extra_info = {}

        extra_info["file_path"] = str(file_path)
        extra_info["should_split"] = should_split

        documents = []
        for doc in docs:
            documents.append(
                Document(
                    document=doc.get_text().encode(encoding),
                    metadata=dict(
                        source=f"{file_name}: Page {doc.number + 1}",
                        **extra_info,
                    ),
                )
            )

        docs.close()
        return documents
