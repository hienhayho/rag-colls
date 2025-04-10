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
        **kwargs,
    ) -> list[Document]:
        """
        Loads data from the specified file path and returns a list of Document objects.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            should_split (bool): Whether to split the data into smaller chunks.
            extra_info (dict | None): Additional information to be passed to the loader.
            **kwargs: Additional keyword arguments to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __str__(self):
        return f"{self.__class__.__name__}"

    async def _aload_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
        **kwargs,
    ) -> list[Document]:
        """
        Asynchronously loads data from the specified file path and returns a list of Document objects.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            should_split (bool): Whether to split the data into smaller chunks.
            extra_info (dict | None): Additional information to be passed to the loader.
            **kwargs: Additional keyword arguments to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        return await asyncio.to_thread(
            self._load_data, file_path, should_split, extra_info, **kwargs
        )

    def load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
        **kwargs,
    ) -> list[Document]:
        """
        Loads data from the specified file path and returns a list of Document objects.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            should_split (bool): Whether to split the data into smaller chunks.
            extra_info (dict | None): Additional information to be passed to the loader.
            **kwargs: Additional keyword arguments to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        documents = self._load_data(file_path, should_split, extra_info, **kwargs)

        for doc in documents:
            assert doc.metadata.get("should_split") is not None, (
                f"Document metadata should contain 'should_split' key. Got: {doc.metadata}"
            )

        return documents

    async def aload_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
        **kwargs,
    ) -> list[Document]:
        """
        Asynchronously loads data from the specified file path and returns a list of Document objects.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            should_split (bool): Whether to split the data into smaller chunks.
            extra_info (dict | None): Additional information to be passed to the loader.
            **kwargs: Additional keyword arguments to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        documents = await self._aload_data(
            file_path, should_split, extra_info, **kwargs
        )

        for doc in documents:
            assert doc.metadata.get("should_split") is not None, (
                f"Document metadata should contain 'should_split' key. Got: {doc.metadata}"
            )

        return documents
