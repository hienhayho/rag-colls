from loguru import logger
from abc import abstractmethod

from rag_colls.types.retriever import RetrieverResult, RetrieverIngestInput
from rag_colls.core.base.retrievers.base_retriever_provider import BaseRetrieverProvider


class BaseVectorDatabase(BaseRetrieverProvider):
    """
    Abstract base class for vector databases.
    """

    @abstractmethod
    def _check_collection_exists(self, collection_name: str) -> bool:
        """
        Check the connection to the vector database.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _create_collection(self, collection_name: str, **kwargs):
        """
        Create a collection in the vector database.

        Args:
            collection_name (str): The name of the collection to be created.
            **kwargs: Additional keyword arguments for the collection creation.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def __str__(self):
        """
        Return a string representation of the vector database.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _add_documents(self, documents: list[RetrieverIngestInput], **kwargs):
        """
        Add documents to the vector database.

        Args:
            documents (list[RetrieverIngestInput]): List of documents to be added.
            **kwargs: Additional keyword arguments for the add operation.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _delete_documents(self, document_ids: list[str], **kwargs):
        """
        Delete documents from the vector database.

        Args:
            document_ids (list[str]): List of document IDs to be deleted.
            **kwargs: Additional keyword arguments for the delete operation.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _search(
        self, query_embedding: list[float], top_k: int = 5, **kwargs
    ) -> list[RetrieverResult]:
        """
        Retrieve documents from the vector database based on the query.

        Args:
            query_embedding (list[float]): The embedding of the query to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the retrieval operation.

        Returns:
            list[RetrieverResult]: List of retrieved documents with their metadata.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def _asearch(
        self, query_embedding: list[float], top_k: int = 5, **kwargs
    ) -> list[RetrieverResult]:
        """
        Asynchronously retrieve documents from the vector database based on the query.

        Args:
            query_embedding (list[float]): The embedding of the query to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the retrieval operation.

        Returns:
            list[RetrieverResult]: List of retrieved documents with their metadata.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def search(
        self, query_embedding: list[float], top_k: int = 5, **kwargs
    ) -> list[RetrieverResult]:
        """
        Retrieve documents from the vector database based on the query.

        Args:
            query_embedding (list[float]): The embedding of the query to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the retrieval operation.

        Returns:
            list[RetrieverResult]: List of retrieved documents with their metadata.
        """
        return self._search(query_embedding, top_k, **kwargs)

    async def asearch(
        self, query_embedding: list[float], top_k: int = 5, **kwargs
    ) -> list[RetrieverResult]:
        """
        Asynchronously retrieve documents from the vector database based on the query.

        Args:
            query_embedding (list[float]): The embedding of the query to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the retrieval operation.

        Returns:
            list[RetrieverResult]: List of retrieved documents with their metadata.
        """
        return await self._asearch(query_embedding, top_k, **kwargs)

    def add_documents(self, documents: list[RetrieverIngestInput], **kwargs):
        """
        Add documents to the vector database.

        Args:
            documents (list[RetrieverIngestInput]): List of documents to be added.
            **kwargs: Additional keyword arguments for the add operation.
        """
        if not self._check_collection_exists(self.collection_name):
            logger.warning(
                f"Collection {self.collection_name} does not exist. Creating collection."
            )
            self._create_collection(self.collection_name, **kwargs)

        self._add_documents(documents, **kwargs)

    def delete_documents(self, document_ids: list[str], **kwargs):
        """
        Delete documents from the vector database.

        Args:
            document_ids (list[str]): List of document IDs to be deleted.
            **kwargs: Additional keyword arguments for the delete operation.
        """
        if not self._check_collection_exists(self.collection_name):
            logger.warning(
                f"Collection {self.collection_name} does not exist. Cannot delete documents."
            )
            return

        self._delete_documents(document_ids, **kwargs)

    def clean_resource(self):
        """
        Clean up the vector database resources.
        """
        return self._clean_resource()
