from abc import ABC, abstractmethod
from rag_colls.types.llm import LLMOutput


class BaseRAG(ABC):
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
    def _search(self, query: str, **kwargs) -> LLMOutput:
        """
        Search for the most relevant documents based on the query.

        Args:
            query (str): The query to search for.
            **kwargs: Additional keyword arguments for the search operation.

        Returns:
            LLMOutput: The response from the LLM.
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

    def search(self, query: str, **kwargs) -> LLMOutput:
        """
        Search for the most relevant documents based on the query.

        Args:
            query (str): The query to search for.
            **kwargs: Additional keyword arguments for the search operation.

        Returns:
            LLMOutput: The response from the LLM.
        """
        return self._search(query, **kwargs)
