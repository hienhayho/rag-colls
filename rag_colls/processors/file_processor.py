import warnings

from typing import Any
from pathlib import Path
from loguru import logger
from multiprocessing import Pool
from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader


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

    def __str__(self):
        return "FileProcessor"

    def _get_default_processors(self) -> dict[str, BaseReader]:
        """
        Initialize default file processors.

        Returns:
            dict[str, BaseReader]: A dictionary of default file processors.
        """
        logger.info("Initializing default file processors ...")
        from .readers.pdf import PyMuPDFReader
        from .readers.csv import CSVReader
        from .readers.docx import DocxReader
        from .readers.txt import TxtReader
        from .readers.json import JSONReader
        from .readers.html import HTMLReader
        from .readers.excel import ExcelReader

        return {
            ".pdf": PyMuPDFReader(),
            ".csv": CSVReader(),
            ".docx": DocxReader(),
            ".txt": TxtReader(),
            ".json": JSONReader(),
            ".html": HTMLReader(),
            ".xlsx": ExcelReader(),
            ".xls": ExcelReader(),
        }

    def _get_all_file_paths(
        self,
        file_or_folder_paths: list[str | Path],
        should_splits: list[bool],
        extra_infos: list[dict],
    ) -> list[Path]:
        """
        Get all valid file paths from the given list of file or folder paths.

        Args:
            file_or_folder_paths (list[str | Path]): List of file or folder paths.
            should_splits (list[bool]): List of boolean values indicating whether to split the files.
            extra_infos (list[dict]): List of dictionaries containing extra information for each file.

        Returns:
            list[Path]: List of valid file paths.
        """
        all_file_paths = []
        new_should_splits = []
        new_extra_infos = []

        for path, should_split, extra_info in zip(
            file_or_folder_paths, should_splits, extra_infos
        ):
            if isinstance(path, str):
                path = Path(path)
            if isinstance(path, Path):
                if path.is_dir():
                    files = [
                        file
                        for file in path.glob("**/*")
                        if file.is_file() and file.suffix in self.processors
                    ]
                    all_file_paths.extend(files)
                    new_should_splits.extend([should_split] * len(files))
                    new_extra_infos.extend([extra_info] * len(files))

                elif path.is_file() and path.suffix in self.processors:
                    all_file_paths.append(path)
                    new_should_splits.append(should_split)
                    new_extra_infos.append(extra_info)

                else:
                    warnings.warn(f"Invalid file_paths: {path}")
        return (
            [str(file) for file in all_file_paths],
            new_should_splits,
            new_extra_infos,
        )

    def load_data(
        self,
        file_or_folder_paths: list[str | Path],
        should_splits: list[bool] | None = None,
        extra_infos: list[dict] | None = None,
        num_workers: int = 1,
    ) -> list[Document]:
        """
        Load data from the given list of file or folder paths.

        Args:
            file_or_folder_paths (list[str | Path]): List of file or folder paths.
            should_splits (list[bool] | None): List of boolean values indicating whether to split the files.
            extra_infos (list[dict] | None): List of dictionaries containing extra information for each file.
            num_workers (int): Number of worker processes to use for multiprocessing.

        Returns:
            list[Document]: List of processed documents.
        """
        logger.info(f"Processing {len(file_or_folder_paths)} paths ...")

        should_splits = should_splits or [True] * len(file_or_folder_paths)
        extra_infos = extra_infos or [None] * len(file_or_folder_paths)

        assert len(file_or_folder_paths) == len(should_splits), (
            "file_or_folder_paths and should_splits must have the same length."
        )

        assert len(file_or_folder_paths) == len(extra_infos), (
            "file_or_folder_paths and extra_infos must have the same length."
        )

        file_paths, should_splits, extra_infos = self._get_all_file_paths(
            file_or_folder_paths, should_splits, extra_infos
        )

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
        file_or_folder_paths: list[str | Path],
        should_splits: list[bool] | None = None,
        extra_infos: list[dict] | None = None,
        max_workers: int = 1,
    ) -> list[Document]:
        """
        Asynchronous version of load_data.

        Args:
            file_or_folder_paths (list[str | Path]): List of file or folder paths.
            should_splits (list[bool] | None): List of boolean values indicating whether to split the files.
            extra_infos (list[dict] | None): List of dictionaries containing extra information for each file.
            max_workers (int): Number of worker processes to use for multiprocessing.

        Returns:
            list[Document]: List of processed documents.
        """
        logger.info(f"Processing {len(file_or_folder_paths)} files asynchronously ...")

        should_splits = should_splits or [True] * len(file_or_folder_paths)
        extra_infos = extra_infos or [None] * len(file_or_folder_paths)

        assert len(file_or_folder_paths) == len(should_splits), (
            "file_or_folder_paths and should_splits must have the same length."
        )

        assert len(file_or_folder_paths) == len(extra_infos), (
            "file_or_folder_paths and extra_infos must have the same length."
        )

        file_paths, should_splits, extra_infos = self._get_all_file_paths(
            file_or_folder_paths, should_splits, extra_infos
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
