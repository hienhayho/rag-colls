import polars as pl
from loguru import logger
from abc import ABC, abstractmethod
from rag_colls.types.search import SearchOutput
from rag_colls.types.core.document import Document
from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class BaseRAG(ABC):
    llm: BaseCompletionLLM

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
    def _retrieve_db(
        self,
        *,
        query: RetrieverQueryType,
        top_k: int = 5,
        **kwargs,
    ) -> list[RetrieverResult]:
        raise NotImplementedError(
            "Retrieving documents from the database process is not implemented."
        )

    @abstractmethod
    def _search(self, *, query: RetrieverQueryType, **kwargs) -> SearchOutput:
        """
        Search for the most relevant documents based on the query.

        Args:
            query (RetrieverQueryType): The query to search for.
            **kwargs: Additional keyword arguments for the search operation.

        Returns:
            SearchOutput: The response from the LLM or a tuple of the response and retrieved results.
        """
        raise NotImplementedError("Searching documents process is not implemented.")

    @abstractmethod
    def _get_chunks(self, file_or_folder_paths: list[str], **kwargs) -> list[Document]:
        """
        Get documents from the documents.

        Args:
            file_or_folder_paths (list[str]): List of file paths or folders to be chunked.
            **kwargs: Additional keyword arguments for the chunking process.

        Returns:
            list[Document]: List of Document objects representing the chunked documents.
        """
        raise NotImplementedError("Chunking documents process is not implemented.")

    @abstractmethod
    def _ingest_db_from_chunks(self, chunks: list[Document], **kwargs):
        """
        Ingest documents into the vector database.

        Args:
            chunks (list[Document]): List of chunks objects to be ingested.
            **kwargs: Additional keyword arguments for the ingestion process.
        """
        raise NotImplementedError("Ingesting documents process is not implemented.")

    def ingest_db(
        self,
        file_or_folder_paths: list[str],
        save_chunk: bool = False,
        output_path: str | None = None,
        **kwargs,
    ):
        """
        Ingest documents into the vector database.

        Args:
            file_or_folder_paths (list[str]): List of file paths or folders to be ingested.
            **kwargs: Additional keyword arguments for the ingestion process.
        """
        chunks = self._get_chunks(file_or_folder_paths=file_or_folder_paths, **kwargs)
        if save_chunk:
            if output_path is None:
                raise ValueError(
                    "output_path must be provided when save_chunk is True."
                )
            self._save_chunks(chunks=chunks, output_path=output_path)

        return self._ingest_db_from_chunks(chunks=chunks, **kwargs)

    def _save_chunks(self, chunks: list[Document], output_path: str):
        """
        Save chunks to the vector database.

        Args:
            chunks (ChunksType): List of chunks objects to be saved.
            output_path (str): Path to save the chunks. (JSONL)
        """
        data = [
            {"id": chunk.id, "document": chunk.document, "metadata": chunk.metadata}
            for chunk in chunks
        ]

        df = pl.DataFrame(data)
        df.write_ndjson(output_path)
        logger.success(f"Chunks saved to: {output_path}")

    def ingest_db_from_chunks(self, chunks: list[Document], **kwargs):
        """
        Ingest documents into the vector database.

        Args:
            chunks (ChunksType): List of chunks objects to be ingested.
            **kwargs: Additional keyword arguments for the ingestion process.
        """
        return self._ingest_db_from_chunks(chunks=chunks, **kwargs)

    def retrieve_db(
        self,
        *,
        query: RetrieverQueryType,
        top_k: int = 5,
        **kwargs,
    ) -> list[RetrieverResult]:
        """
        Retrieve documents from the vector database.

        Args:
            query (RetrieverQueryType): The query to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the retrieval process.

        Returns:
            list[RetrieverResult]: List of retrieved documents.
        """
        return self._retrieve_db(query=query, top_k=top_k, **kwargs)

    def search(self, *, query: RetrieverQueryType, **kwargs) -> SearchOutput:
        """
        Search for the most relevant documents based on the query.

        Args:
            query (RetrieverQueryType): The query to search for.
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
