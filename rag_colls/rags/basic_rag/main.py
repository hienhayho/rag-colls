from rag_colls.core.base.readers.base import BaseReader
from rag_colls.core.base.chunkers.base import BaseChunker
from rag_colls.core.base.embeddings.base import BaseEmbedding
from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.core.base.database.vector_database import BaseVectorDatabase

from rag_colls.prompts.q_a import Q_A_PROMPT
from rag_colls.types.llm import Message, LLMOutput
from rag_colls.core.settings import GlobalSettings
from rag_colls.types.retriever import RetrieverIngestInput
from rag_colls.processors.file_processor import FileProcessor


class BasicRAG:
    """
    Wrapper class for a basic RAG (Retrieval-Augmented Generation) system. This class integrates a vector database, chunker, LLM, and embedding model to perform semantic search.

    This is helpful for anyone begin starting with RAG and understanding how to use it.
    """

    def __init__(
        self,
        vector_database: BaseVectorDatabase,
        chunker: BaseChunker,
        llm: BaseCompletionLLM | None = None,
        embed_model: BaseEmbedding | None = None,
        processors: dict[str, BaseReader] | None = None,
    ):
        """
        Initialize the BasicRAG class.

        Args:
            vector_database (BaseVectorDatabase): The vector database to use for storing and retrieving documents.
            chunker (BaseChunker): The chunker to use for splitting documents into smaller chunks.
            llm (BaseCompletionLLM, optional): The LLM to use for generating responses. Defaults to `None`.
            embed_model (BaseEmbedding, optional): The embedding model to use for generating embeddings. Defaults to `None`.
            processors (dict[str, BaseReader], optional): A dictionary of processors for different file types. Defaults to `None`.
        """
        self.vector_database = vector_database
        self.chunker = chunker
        self.processor = FileProcessor(processors=processors)
        self.embed_model = embed_model or GlobalSettings.embed_model
        self.llm = llm or GlobalSettings.completion_llm

    def ingest_db(self, file_paths: list[str], batch_embedding: int = 100):
        """
        Ingest documents into the vector database.

        Args:
            file_paths (list[str]): List of file paths to be ingested.
            batch_embedding (int): Batch size for embedding documents.
        """
        documents = self.processor.load_data(file_paths=file_paths)

        chunks = self.chunker.chunk(documents=documents)

        embeddings = self.embed_model.get_batch_document_embedding(
            documents=chunks, batch_size=batch_embedding
        )

        embeded_chunks = [
            RetrieverIngestInput(
                id=doc.id,
                document=doc.document,
                embedding=e.embedding,
                metadata=doc.metadata,
            )
            for doc, e in zip(chunks, embeddings)
        ]

        self.vector_database.add_documents(
            documents=embeded_chunks,
        )

    def search(self, query: str, top_k: int = 5, **kwargs) -> LLMOutput:
        """
        Search for the most relevant documents based on the query.

        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the search operation.

        Returns:
            LLMOutput: The response from the LLM.
        """
        embedding = self.embed_model.get_query_embedding(query=query)

        results = self.vector_database.search(
            query_embedding=embedding.embedding,
            k=top_k,
        )

        contexts = ""

        for result in results:
            contexts += f"{result.document} ============ \n"

        messages = [
            Message(
                role="user", content=Q_A_PROMPT.format(context=contexts, question=query)
            )
        ]

        response = self.llm.complete(messages=messages)

        return response
