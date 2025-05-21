from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag_colls.rags.base import BaseRAG
from rag_colls.core.base.chunkers.base import BaseChunker
from rag_colls.core.base.rerankers.base import BaseReranker
from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.core.base.embeddings.base import BaseEmbedding
from rag_colls.core.base.database.bm25 import BaseBM25RetrieverProvider
from rag_colls.core.base.database.vector_database import BaseVectorDatabase
from rag_colls.core.utils import run_fuction_return_time

from rag_colls.types.llm import Message
from rag_colls.types.search import SearchOutput
from rag_colls.types.core.document import Document
from rag_colls.core.settings import GlobalSettings
from rag_colls.processors.file_processor import FileProcessor
from rag_colls.retrievers.bm25_retriever import BM25Retriever
from rag_colls.retrievers.hybrid_retriever import HybridRetriever
from rag_colls.types.retriever import (
    RetrieverIngestInput,
    RetrieverQueryType,
    RetrieverResult,
)
from rag_colls.retrievers.vector_database_retriever import VectorDatabaseRetriever

from .utils import get_contextual_boost_document
from .prompt import (
    CONTEXTUAL_BOOST_SYSTEM_PROMPT,
    get_prompt,
    gen_data_prompt,
    PromptModeEnum,
)


class RAFT(BaseRAG):
    """"""

    def __init__(
        self,
        *,
        vector_database: BaseVectorDatabase,
        bm25: BaseBM25RetrieverProvider,
        reranker: BaseReranker,
        chunker: BaseChunker,
        llm: BaseCompletionLLM | None = None,
        embed_model: BaseEmbedding | None = None,
        processor: FileProcessor | None = None,
        gen_contextual_system_prompt: str = CONTEXTUAL_BOOST_SYSTEM_PROMPT,
    ):
        self.vector_database = vector_database
        self.bm25 = bm25
        self.reranker = reranker
        self.chunker = chunker
        self.processor = processor or FileProcessor()

        self.embed_model = embed_model or GlobalSettings.embed_model
        self.llm = llm or GlobalSettings.completion_llm

        self.semantic_retriever = VectorDatabaseRetriever.from_vector_db(
            vector_db=vector_database, embed_model=self.embed_model
        )
        self.bm25_retriever = BM25Retriever.from_bm25(bm25=self.bm25)

        self.retriever = HybridRetriever(
            retrievers=[self.semantic_retriever, self.bm25_retriever],
            reranker=self.reranker,
        )
        self.gen_contextual_system_prompt = gen_contextual_system_prompt

    def _get_metadata(self):
        """
        Get the metadata of the Contextual RAG.

        Returns:
            dict: Metadata of the Contextual RAG.
        """
        return {
            "vector_database": str(self.vector_database),
            "bm25": str(self.bm25),
            "chunker": str(self.chunker),
            "llm": str(self.llm),
            "embed_model": str(self.embed_model),
            "processor": str(self.processor),
            "reranker": str(self.reranker),
        }

    def _clean_resource(self):
        """
        Clean the retriever resource.
        """
        self.retriever.clean_resource()

    def _get_chunks(self, file_or_folder_paths: list[str], **kwargs):
        """
        Get documents from the file or folder paths.

        Args:
            file_or_folder_paths (list[str]): List of file paths or folders to be chunked.
            **kwargs: Additional keyword arguments for the chunking process.

        Returns:
            list[Document]: List of Document objects representing the chunked documents.
        """
        documents = self.processor.load_data(file_or_folder_paths=file_or_folder_paths)

        chunks = self.chunker.chunk(
            documents=documents,
            **kwargs,
        )

        num_workers = kwargs.get("num_workers", 4)

        new_chunks: list[Document] = []
        for chunk in chunks:
            new_chunks.extend(chunk)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    get_contextual_boost_document,
                    document,
                    self.llm,
                    self.gen_contextual_system_prompt,
                )
                for document in new_chunks
            }

            contextual_boost_chunks: list[Document] = []
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating contextual chunks ...",
            ):
                contextual_boost_chunks.append(future.result())

        return contextual_boost_chunks

    def _ingest_db_from_chunks(
        self,
        chunks: list[Document],
        batch_embedding: int = 100,
        **kwargs,
    ) -> None:
        """
        Ingest documents into the Contextual RAG database.

        Args:
            chunks (ChunksType): List of chunks to be ingested.
            batch_embedding (int): Batch size for embedding documents.
            num_workers (int): Number of workers for parallel processing.
            **kwargs: Additional keyword arguments for the ingestion process.
        """

        embeddings = self.embed_model.get_batch_document_embedding(
            documents=chunks, batch_size=batch_embedding, **kwargs
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
        self.bm25.add_documents(
            documents=embeded_chunks,
        )

    def _retrieve_db(
        self,
        *,
        query: RetrieverQueryType,
        top_k: int = 5,
        **kwargs,
    ) -> list[RetrieverResult]:
        """
        Retrieve documents from the database.

        Args:
            query (RetrieverQueryType): The query to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the retrieval process.

        Returns:
            list[Document]: A list of retrieved documents.
        """
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            **kwargs,
        )
        return results

    def _search(
        self,
        *,
        query: RetrieverQueryType,
        top_k: int = 5,
        prompt_mode: PromptModeEnum = PromptModeEnum.JSON,
        **kwargs,
    ) -> SearchOutput:
        """
        Search with Contextual RAG.

        Args:
            query (RetrieverQueryType): The query to search for.
            return_retrieved_result (bool): Whether to return the retrieved result.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the search operation.

        Returns:
            SearchOutput: The response from the LLM or a tuple of the response and retrieved results.
        """

        retrieved_time, search_results = run_fuction_return_time(
            self.retrieve_db,
            query=query,
            top_k=top_k,
            **kwargs,
        )

        system_prompt, user_prompt = get_prompt(
            prompt=gen_data_prompt,
            mode=prompt_mode,
        )

        contexts = "\n\n".join(
            f"<DOCUMENT>{result.document}</DOCUMENT>" for result in search_results
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(
                role="user", content=user_prompt.format(context=contexts, query=query)
            ),
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
