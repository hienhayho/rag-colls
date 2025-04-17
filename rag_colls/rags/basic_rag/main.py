from rag_colls.rags.base import BaseRAG
from rag_colls.core.base.chunkers.base import BaseChunker
from rag_colls.core.base.embeddings.base import BaseEmbedding
from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.core.base.database.vector_database import BaseVectorDatabase
from rag_colls.core.utils import run_fuction_return_time


from rag_colls.types.llm import Message
from rag_colls.prompts.q_a import Q_A_PROMPT
from rag_colls.types.search import SearchOutput
from rag_colls.core.settings import GlobalSettings
from rag_colls.types.retriever import RetrieverIngestInput
from rag_colls.processors.file_processor import FileProcessor
from rag_colls.retrievers.vector_database_retriever import VectorDatabaseRetriever


class BasicRAG(BaseRAG):
    """
    Wrapper class for a basic RAG (Retrieval-Augmented Generation) system. This class integrates a vector database, chunker, LLM, and embedding model to perform semantic search.

    This is helpful for anyone begin starting with RAG and understanding how to use it.
    """

    def __init__(
        self,
        *,
        vector_database: BaseVectorDatabase,
        chunker: BaseChunker | None = None,
        llm: BaseCompletionLLM | None = None,
        embed_model: BaseEmbedding | None = None,
        processor: FileProcessor | None = None,
    ):
        """
        Initialize the BasicRAG class.

        Args:
            vector_database (BaseVectorDatabase): The vector database to use for storing and retrieving documents.
            chunker (BaseChunker): The chunker to use for splitting documents into smaller chunks.
            llm (BaseCompletionLLM, optional): The LLM to use for generating responses. Defaults to `None`.
            embed_model (BaseEmbedding, optional): The embedding model to use for generating embeddings. Defaults to `None`.
            processor: (FileProcessor, optional): The processor to use for loading and processing documents. Defaults to `None`.
        """
        self.vector_database = vector_database
        self.chunker = chunker
        self.processor = processor or FileProcessor()
        self.embed_model = embed_model or GlobalSettings.embed_model
        self.llm = llm or GlobalSettings.completion_llm

        self.retriever = VectorDatabaseRetriever.from_vector_db(
            vector_db=vector_database, embed_model=self.embed_model
        )

    def _ingest_db(self, file_or_folder_paths: list[str], batch_embedding: int = 5):
        """
        Ingest documents into the vector database.

        Args:
            file_paths (list[str]): List of file paths to be ingested.
            batch_embedding (int): Batch size for embedding documents.
        """
        documents = self.processor.load_data(file_or_folder_paths=file_or_folder_paths)

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

    def _clean_resource(self):
        """
        Clean the retriever resource.
        """
        self.vector_database.clean_resource()

    def _get_metadata(self) -> dict:
        """
        Get metadata from the vector database.

        Returns:
            dict: Metadata of the RAG instance.
        """
        return {
            "vector_database": str(self.vector_database),
            "chunker": str(self.chunker),
            "processor": str(self.processor),
            "embed_model": str(self.embed_model),
            "llm": str(self.llm),
        }

    def _search(
        self,
        *,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> SearchOutput:
        """
        Search for the most relevant documents based on the query.

        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the search operation.

        Returns:
            LLMOutput: The response from the LLM.
        """

        retrieved_time, search_results = run_fuction_return_time(
            self.retriever.retrieve,
            query=query,
            top_k=top_k,
            **kwargs,
        )

        contexts = "\n ============ \n".join(
            result.document for result in search_results
        )

        messages = [
            Message(
                role="user", content=Q_A_PROMPT.format(context=contexts, question=query)
            )
        ]

        generation_time, response = run_fuction_return_time(
            self.llm.complete,
            messages=messages,
        )

        return SearchOutput(
            content=response.content,
            usage=response.usage,
            retrieved_results=search_results,
            retrieved_time=retrieved_time,
            generation_time=generation_time,
        )
