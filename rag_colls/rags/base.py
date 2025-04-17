from abc import ABC, abstractmethod
from rag_colls.types.search import SearchOutput
from rag_colls.core.base.llms.base import BaseCompletionLLM


class BaseRAG(ABC):
    llm: BaseCompletionLLM

    @abstractmethod
    def _ingest_db(self, file_or_folder_paths: list[str], **kwargs):
        """
        Ingest documents process

        Args:
            file_or_folder_paths (list[str]): List of file paths or folders to be ingested.
            **kwargs: Additional keyword arguments for the ingestion process.
        """
        raise NotImplementedError("Ingesting documents process is not implemented.")

    @abstractmethod
    def _clean_resource(self):
        """
        Clean the retriever resource.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _get_metadata(self) -> dict:
        """
        Get metadata from the RAG instance.

        Should return metadata of the instances such as vector database, chunker, and processor, ...
        """
        raise NotImplementedError("Getting metadata process is not implemented.")

    @abstractmethod
    def _search(self, *, query: str, **kwargs) -> SearchOutput:
        """
        Search for the most relevant documents based on the query.

        Args:
            query (str): The query to search for.
            **kwargs: Additional keyword arguments for the search operation.

        Returns:
            SearchOutput: The response from the LLM or a tuple of the response and retrieved results.
        """
        raise NotImplementedError("Searching documents process is not implemented.")

    def ingest_db(self, file_or_folder_paths: list[str], **kwargs):
        """
        Ingest documents into the vector database.

        Args:
            file_or_folder_paths (list[str]): List of file paths or folders to be ingested.
            **kwargs: Additional keyword arguments for the ingestion process.
        """
        return self._ingest_db(file_or_folder_paths=file_or_folder_paths, **kwargs)

    def search(self, *, query: str, **kwargs) -> SearchOutput:
        """
        Search for the most relevant documents based on the query.

        Args:
            query (str): The query to search for.
            return_retrieved_result (bool): Whether to return the retrieved result.
            **kwargs: Additional keyword arguments for the search operation.

        Returns:
            SearchOutput: The response from the LLM or a tuple of the response and retrieved results.
        """
        return self._search(query=query, **kwargs)

    def clean_resource(self):
        """
        Clean the retriever resource.
        """
        return self._clean_resource()

    def get_metadata(self):
        """
        Get metadata from the vector database.

        Args:
            **kwargs: Additional keyword arguments for the metadata retrieval process.
        """
        return self._get_metadata()

    def get_llm(self) -> BaseCompletionLLM:
        """
        Get the LLM instance.

        Returns:
            BaseCompletionLLM: The LLM instance.
        """
        return self.llm
