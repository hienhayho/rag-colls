try:
    from megaparse import MegaParse
    from megaparse.parser.megaparse_vision import MegaParseVision
except ImportError:
    raise ImportError(
        "The 'magaparse' package is required for this module. "
        "Please install it using 'pip install megaparse'."
    )

from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader

load_dotenv()


class MegaParseReader(BaseReader):
    """
    Reader using the MegaParse library to parse documents.
    """

    def __init__(
        self, megaparse_converter: MegaParse | MegaParseVision | None = None
    ) -> None:
        if megaparse_converter is None:
            logger.info(
                "No MegaParse instance provided, initializing default MegaParse ..."
            )
            megaparse_converter = MegaParse()
        self.megaparse_converter = megaparse_converter
        logger.info("MegaParseReader initialized !!!")

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

        if isinstance(self.megaparse_converter, MegaParse):
            result = self.megaparse_converter.load(file_path=str(file_path))
        elif isinstance(self.megaparse_converter, MegaParseVision):
            document = self.megaparse_converter.convert(file_path=str(file_path))
            document.clean()
            result = str(document)
        else:
            raise ValueError("Invalid MegaParse converter instance.")

        return [
            Document(
                document=result,
                metadata={
                    **extra_info,
                    "source": file_name,
                },
            )
        ]
