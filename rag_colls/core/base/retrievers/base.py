from abc import ABC, abstractmethod
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class BaseRetriever(ABC):
    """
    Base class for all retrievers.
    """

    @abstractmethod
    def _retrieve(self, query: RetrieverQueryType, **kwargs) -> list[RetrieverResult]:
        """
        Retrieve documents based on the query.

        Args:
            query (RetrieverQueryType): The query to search for.
            **kwargs: Additional keyword arguments for the retrieval process.

        Returns:
            list[RetrieverResult]: A list of retrieved documents.
        """
        raise NotImplementedError("Retrieving documents process is not implemented.")

    def retrieve(self, query: RetrieverQueryType, **kwargs) -> list[RetrieverResult]:
        """
        Retrieve documents based on the query.

        Args:
            query (RetrieverQueryType): The query to search for.
            **kwargs: Additional keyword arguments for the retrieval process.

        Returns:
            list[RetrieverResult]: A list of retrieved documents.
        """
        return self._retrieve(query, **kwargs)
