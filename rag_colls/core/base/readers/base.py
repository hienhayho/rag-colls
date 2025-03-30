import asyncio
from pathlib import Path
from abc import ABC, abstractmethod

from rag_colls.types.core.document import Document


class BaseReader(ABC):
    @abstractmethod
    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Loads data from the specified file path and returns a list of Document objects.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            should_split (bool): Whether to split the data into smaller chunks.
            extra_info (dict | None): Additional information to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    async def _aload_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Asynchronously loads data from the specified file path and returns a list of Document objects.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            extra_info (dict | None): Additional information to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        return await asyncio.to_thread(
            self._load_data, file_path, should_split, extra_info
        )

    def load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Loads data from the specified file path and returns a list of Document objects.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            should_split (bool): Whether to split the data into smaller chunks.
            extra_info (dict | None): Additional information to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        return self._load_data(file_path, should_split, extra_info)

    async def aload_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Asynchronously loads data from the specified file path and returns a list of Document objects.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            should_split (bool): Whether to split the data into smaller chunks.
            extra_info (dict | None): Additional information to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        return await self._aload_data(file_path, should_split, extra_info)
