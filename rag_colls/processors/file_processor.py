from typing import Any
from pathlib import Path
from multiprocessing import Pool

from rag_colls.types.core.document import Document
from rag_colls.core.settings import GlobalSettings
from rag_colls.core.base.readers.base import BaseReader

logger = GlobalSettings.logger


def process_file_worker(
    args: tuple[str | Path, bool, dict[str, Any], dict[str, BaseReader]],
) -> list[Document]:
    """
    Worker function for multiprocessing.

    Args:
        args (tuple): (file_path, should_split, extra_info, processors)

    Returns:
        list[Document]: List of processed documents.
    """
    file_path, should_split, extra_info, processors = args
    ext = Path(file_path).suffix.lower()

    if ext not in processors:
        raise ValueError(f"No processor found for file type: {ext}")

    reader = processors[ext]
    return reader.load_data(
        file_path=file_path, should_split=should_split, extra_info=extra_info
    )


async def process_file_worker_async(
    args: tuple[str | Path, bool, dict[str, Any], dict[str, BaseReader]],
) -> list[Document]:
    """
    Asynchronous worker function for processing files.
    Args:
        args (tuple): (file_path, should_split, extra_info, processors)
    Returns:
        list[Document]: List of processed documents.
    """
    file_path, should_split, extra_info, processors = args
    ext = Path(file_path).suffix.lower()

    if ext not in processors:
        raise ValueError(f"No processor found for file type: {ext}")

    reader = processors[ext]
    return await reader.aload_data(
        file_path=file_path, should_split=should_split, extra_info=extra_info
    )


class FileProcessor:
    def __init__(
        self,
        processors: dict[str, BaseReader] | None = None,
        merge_with_default_processors: bool = False,
    ):
        """
        Initialize the FileProcessor with a dictionary of file type processors.
        """
        self.processors = processors or {}

        if not processors:
            logger.info("No processors provided. Using default processors ...")
            merge_with_default_processors = True

        if merge_with_default_processors:
            default_processors = self._get_default_processors()
            for ext, processor in default_processors.items():
                if ext not in self.processors:
                    self.processors[ext] = processor

    def _get_default_processors(self) -> dict[str, BaseReader]:
        logger.info("Initializing default file processors ...")
        from .readers.pdf import PyMuPDFReader

        return {".pdf": PyMuPDFReader()}

    def load_data(
        self,
        file_paths: list[str | Path],
        should_splits: list[bool] | None = None,
        extra_infos: list[dict] | None = None,
        num_workers: int = 1,
    ) -> list[Document]:
        logger.info(f"Processing {len(file_paths)} files ...")

        should_splits = should_splits or [True] * len(file_paths)
        extra_infos = extra_infos or [None] * len(file_paths)

        assert len(file_paths) == len(should_splits), (
            "file_paths and should_splits must have the same length."
        )
        assert len(file_paths) == len(extra_infos), (
            "file_paths and extra_infos must have the same length."
        )
        assert num_workers > 0, "num_workers must be greater than 0."
        assert isinstance(num_workers, int), "num_workers must be an integer."

        args_list = [
            (file_paths[i], should_splits[i], extra_infos[i], self.processors)
            for i in range(len(file_paths))
        ]

        documents = []

        if num_workers == 1:
            for args in args_list:
                documents.extend(process_file_worker(args))
        else:
            with Pool(num_workers) as pool:
                results = pool.map(process_file_worker, args_list)
                for result in results:
                    documents.extend(result)

        logger.info(f"Get {len(documents)} documents.")
        return documents

    async def aload_data(
        self,
        file_paths: list[str | Path],
        should_splits: list[bool] | None = None,
        extra_infos: list[dict] | None = None,
        max_workers: int = 1,
    ) -> list[Document]:
        """
        Asynchronous version of load_data.
        """
        logger.info(f"Processing {len(file_paths)} files asynchronously ...")

        should_splits = should_splits or [True] * len(file_paths)
        extra_infos = extra_infos or [None] * len(file_paths)

        assert len(file_paths) == len(should_splits), (
            "file_paths and should_splits must have the same length."
        )
        assert len(file_paths) == len(extra_infos), (
            "file_paths and extra_infos must have the same length."
        )

        args_list = [
            (file_paths[i], should_splits[i], extra_infos[i], self.processors)
            for i in range(len(file_paths))
        ]

        documents = []

        if max_workers == 1:
            for args in args_list:
                documents.extend(await process_file_worker_async(args))
        else:
            with Pool(max_workers) as pool:
                results = await pool.map(process_file_worker_async, args_list)
                for result in results:
                    documents.extend(result)

        logger.info(f"Get {len(documents)} documents.")
        return documents
