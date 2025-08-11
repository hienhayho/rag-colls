try:
    from docling.document_converter import DocumentConverter
except ImportError:
    raise ImportError(
        "The 'docling' package is required for this module. "
        "Please install it using 'pip install docling'."
    )

import json
from enum import Enum
from typing import Any
from pathlib import Path
from loguru import logger

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader


class ExportFormat(Enum):
    """
    Enum for export formats supported by the reader.
    """

    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class DoclingReader(BaseReader):
    """
    Reader using the Docling library.
    """

    def __init__(
        self,
        document_converter: DocumentConverter | None = None,
        export_format: ExportFormat = ExportFormat.MARKDOWN,
        export_kwargs: dict[str, Any] = {},
    ):
        """
        Initialize the DoclingReader.

        Args:
            document_converter (DocumentConverter | None): The document converter to use. If `None`, a default DocumentConverter will be used.
            export_format (ExportFormat): The format to export documents to. Defaults to `ExportFormat.MARKDOWN`.
            export_kwargs (dict[str, Any]): Additional keyword arguments for the export methods.
        """
        if document_converter is None:
            logger.info(
                "No document_converter provided, using default DocumentConverter."
            )

            document_converter = DocumentConverter()
        if not isinstance(document_converter, DocumentConverter):
            raise TypeError(
                "document_converter must be an instance of DocumentConverter."
            )

        self.document_converter = document_converter
        self.export_format = export_format
        self.export_kwargs = export_kwargs

        logger.info("DoclingReader initialized !")

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
        extra_info["should_split"] = should_split

        result = self.document_converter.convert(source=file_path).document
        text: str = ""
        if self.export_format == ExportFormat.MARKDOWN:
            text = result.export_to_markdown(**self.export_kwargs)
        elif self.export_format == ExportFormat.JSON:
            text = json.dumps(result.export_to_dict(**self.export_kwargs))
        elif self.export_format == ExportFormat.HTML:
            text = result.export_to_html(**self.export_kwargs)
        else:
            raise ValueError(f"Unsupported export format: {self.export_format}")

        return [
            Document(
                document=text,
                metadata={
                    "source": str(file_path),
                    **extra_info,
                },
            )
        ]
