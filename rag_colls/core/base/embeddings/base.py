from abc import ABC, abstractmethod
from rag_colls.types.embedding import Embedding
from rag_colls.types.core.document import Document
from tenacity import retry, wait_random_exponential, stop_after_attempt


class BaseEmbedding(ABC):
    @abstractmethod
    def _get_query_embedding(self, query: str, **kwargs) -> Embedding:
        """
        Returns the embedding of the query.

        Args:
            query (str): The query to be embedded.

        Returns:
            Embedding: The embedding object of the query.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _get_document_embedding(self, document: Document, **kwargs) -> Embedding:
        """
        Returns the embedding of the document.

        Args:
            document (Document): The document to be embedded.

        Returns:
            list: The embedding of the document.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _get_batch_query_embedding(
        self, queries: list[str], **kwargs
    ) -> list[Embedding]:
        """
        Returns the embeddings of the queries.

        Args:
            queries (list[str]): The list of queries to be embedded.
            **kwargs: Additional keyword arguments for the embedding function.

        Returns:
            list[Embedding]: The list of embedding objects of the queries.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _get_batch_document_embedding(
        self, documents: list[Document], **kwargs
    ) -> list[Embedding]:
        """
        Returns the embeddings of the documents.

        Args:
            documents (list[Document]): The list of documents to be embedded.

        Returns:
            list[Embedding]: The list of embedding objects of the documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_query_embedding(self, query: str, **kwargs) -> Embedding:
        """
        Returns the embedding of the query.

        Args:
            query (str): The query to be embedded.

        Returns:
            Embedding: The embedding object of the query.
        """
        return self._get_query_embedding(query, **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_document_embedding(self, document: Document, **kwargs) -> Embedding:
        """
        Returns the embedding of the document.

        Args:
            document (Document): The document to be embedded.

        Returns:
            Embedding: The embedding object of the document.
        """
        return self._get_document_embedding(document, **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_batch_query_embedding(
        self, queries: list[str], **kwargs
    ) -> list[Embedding]:
        """
        Returns the embeddings of the queries.

        Args:
            queries (list[str]): The list of queries to be embedded.
            **kwargs: Additional keyword arguments for the embedding function.

        Returns:
            list[Embedding]: The list of embedding objects of the queries.
        """
        return self._get_batch_query_embedding(queries, **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_batch_document_embedding(
        self, documents: list[Document], **kwargs
    ) -> list[Embedding]:
        """
        Returns the embeddings of the documents.

        Args:
            documents (list[Document]): The list of documents to be embedded.
            **kwargs: Additional keyword arguments for the embedding function.

        Returns:
            list[Embedding]: The list of embedding objects of the documents.
        """
        return self._get_batch_document_embedding(documents, **kwargs)
