try:
    from markitdown import MarkItDown
except ImportError as e:
    raise ImportError(
        "The 'markitdown' package is required for this module. "
        "Please install it using 'pip install markitdown[all]'."
    ) from e

from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader

load_dotenv()


class MarkItDownReader(BaseReader):
    """
    Reader using the MarkItDown library.
    """

    def __init__(
        self,
        markitdown_converter: MarkItDown | None = None,
    ):
        """
        Initializes the MarkItDownReader.

        Args:
            markitdown_converter (MarkItDown | None): An existing MarkItDown converter instance.
        """
        if markitdown_converter is None:
            logger.info(
                "No markitdown_converter provided, using default MarkItDown converter."
            )
            markitdown_converter = MarkItDown()

        if not isinstance(markitdown_converter, MarkItDown):
            raise TypeError("markitdown_converter must be an instance of MarkItDown.")

        self.converter = markitdown_converter

        logger.info("MarkItDownReader initialized !")

    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Load data from a file and convert it to a list of Document objects.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        file_name = file_path.name

        if not extra_info:
            extra_info = {}

        extra_info["file_name"] = file_name
        extra_info["file_path"] = str(file_path)
        extra_info["should_split"] = should_split

        result = self.converter.convert(source=str(file_path))
        return [
            Document(
                document=result.markdown,
                metadata={
                    "source": str(file_path),
                    **extra_info,
                },
            )
        ]
