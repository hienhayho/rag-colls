from abc import ABC, abstractmethod
from rag_colls.types.core.document import Document


class BaseChunker(ABC):
    @abstractmethod
    def _chunk(self, documents: list[Document], **kwargs) -> list[Document]:
        """
        Chunk the documents.

        Args:
            documents (list[Document]): List of documents to be chunked.
            **kwargs: Additional keyword arguments for the chunking function.

        Returns:
            list[Document]: List of chunked documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def _achunk(self, documents: list[Document], **kwargs) -> list[Document]:
        """
        Asynchronously chunk the documents.

        Args:
            documents (list[Document]): List of documents to be chunked.
            **kwargs: Additional keyword arguments for the chunking function.

        Returns:
            list[Document]: List of chunked documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def chunk(self, documents: list[Document], **kwargs) -> list[Document]:
        """
        Chunk the documents.

        Args:
            documents (list[Document]): List of documents to be chunked.
            **kwargs: Additional keyword arguments for the chunking function.

        Returns:
            list[Document]: List of chunked documents.
        """
        chunked_documents = []
        preprocessed_documents = []
        for doc in documents:
            if not doc.metadata.get("should_split", True):
                chunked_documents.append(doc)
                continue

            preprocessed_documents.append(doc)

        chunked_documents.extend(self._chunk(preprocessed_documents, **kwargs))

        return chunked_documents

    async def achunk(self, documents: list[Document], **kwargs) -> list[Document]:
        """
        Asynchronously chunk the documents.

        Args:
            documents (list[Document]): List of documents to be chunked.
            **kwargs: Additional keyword arguments for the chunking function.

        Returns:
            list[Document]: List of chunked documents.
        """
        chunked_documents = []
        preprocessed_documents = []
        for doc in documents:
            if not doc.metadata.get("should_split", True):
                chunked_documents.append(doc)
                continue

            preprocessed_documents.append(doc)

        chunked_documents.extend(await self._achunk(preprocessed_documents, **kwargs))

        return chunked_documents
