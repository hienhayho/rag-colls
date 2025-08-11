try:
    from openai import OpenAI
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
        enable_plugins: bool = False,
        docintel_endpoint: str | None = None,
        gpt_model: str | None = None,
    ):
        """
        Initializes the MarkItDownReader.

        Args:
            enable_plugins (bool): Whether to enable plugins for the MarkItDown conversion.
            docintel_endpoint (str | None): The endpoint for document intelligence services.
            gpt_model (str | None): The GPT model to use for conversion. If None, no use LLM.
        """
        self.converter = None
        if gpt_model:
            client = OpenAI()
            self.converter = MarkItDown(
                llm_client=client,
                llm_model=gpt_model,
                enable_plugins=enable_plugins,
                docintel_endpoint=docintel_endpoint,
            )
        else:
            self.converter = MarkItDown(
                enable_plugins=enable_plugins,
                docintel_endpoint=docintel_endpoint,
            )

        assert self.converter is not None, "MarkItDown converter initialization failed."

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
