import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod
from rag_colls.types.retriever import (
    RetrieverResult,
    RetrieverQueryType,
    RetrieverIngestInput,
)


@dataclass
class BaseRetrieverProvider(ABC):
    @abstractmethod
    def _test_connection(self):
        """
        Test the connection to the retriever provider.
        This method should be overridden by subclasses to implement the actual connection test.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _add_documents(
        self,
        documents: list[RetrieverIngestInput],
        **kwargs,
    ):
        """
        Ingest documents into the retriever database.

        Args:
            documents (list[RetrieverIngestInput]): List of documents to ingest.
        """
        raise NotImplementedError(
            "The _add_documents method must be implemented in the subclass."
        )

    @abstractmethod
    def _clean_resource(self):
        """
        Clean the retriever resource.
        This method should be overridden by subclasses to implement the actual cleanup.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _search(self, query: RetrieverQueryType, **kwargs) -> list[RetrieverResult]:
        """
        Search for documents in the retriever provider.
        """
        raise NotImplementedError(
            "The _search method must be implemented in the subclass."
        )

    async def _asearch(
        self,
        query: RetrieverQueryType,
        **kwargs,
    ) -> list[RetrieverResult]:
        """
        Asynchronous search for documents in the retriever provider.
        """
        return await asyncio.to_thread(self._search, query, **kwargs)

    def add_documents(
        self,
        documents: list[RetrieverIngestInput],
        **kwargs,
    ):
        """
        Ingest documents into the retriever database.

        Args:
            documents (list[RetrieverIngestInput]): List of documents to ingest.
        """
        return self._add_documents(documents, **kwargs)

    def search(
        self,
        query: RetrieverQueryType,
        **kwargs,
    ) -> list[RetrieverResult]:
        """
        Search for documents in the retriever provider.
        """
        return self._search(query=query, **kwargs)

    async def asearch(
        self,
        query: RetrieverQueryType,
        **kwargs,
    ) -> list[RetrieverResult]:
        """
        Asynchronous search for documents in the retriever provider.
        """
        return await self._asearch(query=query, **kwargs)

    def clean_resource(self):
        """
        Clean the retriever resource.
        """
        return self._clean_resource()
