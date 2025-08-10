from pathlib import Path
from loguru import logger

try:
    from vllm import LLM
    from ocrflux.inference import parse
except ImportError as e:
    raise ImportError(
        "OCRFluxReader cannot be initialized since not all dependencies are available. "
        "Please install it with 'pip install rag-colls[ocrflux_py]'."
    ) from e

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader


class OCRFluxReader(BaseReader):
    """
    Reader for OCRFlux documents.
    """

    def __init__(
        self,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        download_dir: str = "./model_cache",
    ):
        self.llm = LLM(
            dtype=dtype,
            model="ChatDOC/OCRFlux-3B",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            download_dir=download_dir,
        )
        logger.info("OCRFluxReader initialized !")

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

        extra_info["file_path"] = str(file_path)
        extra_info["should_split"] = should_split

        documents = []
        result = parse(self.llm, str(file_path))

        if result is None:
            logger.error(f"Failed to parse the file: {file_path}")
            return documents

        page_texts: dict[str, str] = result.get("page_texts", {})

        for page_number in page_texts:
            page_text = page_texts[page_number]
            if not page_text.strip():
                continue

            doc = Document(
                document=page_text,
                metadata={
                    "source": f"{file_name}: Page {int(page_number) + 1}",
                    **extra_info,
                },
            )
            documents.append(doc)

        return documents
