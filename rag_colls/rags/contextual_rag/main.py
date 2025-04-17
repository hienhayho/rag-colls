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
from rag_colls.prompts.q_a import Q_A_PROMPT
from rag_colls.types.search import SearchOutput
from rag_colls.types.core.document import Document
from rag_colls.core.settings import GlobalSettings
from rag_colls.core.utils import check_placeholders
from rag_colls.processors.file_processor import FileProcessor
from rag_colls.retrievers.bm25_retriever import BM25Retriever
from rag_colls.types.retriever import RetrieverIngestInput, RetrieverQueryType
from rag_colls.retrievers.vector_database_retriever import VectorDatabaseRetriever

from .utils import gen_contextual_chunk
from .prompt import CONTEXTUAL_PROMPT


class ContextualRAG(BaseRAG):
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
        gen_contextual_prompt_template: str | None = None,
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

        if gen_contextual_prompt_template:
            assert check_placeholders(
                template=gen_contextual_prompt_template,
                placeholders=["CHUNK_CONTENT", "WHOLE_DOCUMENT"],
            ), (
                f"Prompt template must contain the placeholders: {['CHUNK_CONTENT', 'WHOLE_DOCUMENT']}. Example: =======\n{CONTEXTUAL_PROMPT}"
            )

            self.gen_contextual_prompt_template = gen_contextual_prompt_template

        else:
            self.gen_contextual_prompt_template = CONTEXTUAL_PROMPT

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
            "gen_contextual_prompt_template": str(self.gen_contextual_prompt_template),
        }

    def _split_document(self, document: Document, **kwargs) -> list[Document]:
        """
        Split the document into chunks using the chunker.

        Args:
            document (Document): The document to be split.
            **kwargs: Additional arguments for the chunker.

        Returns:
            list[str]: A list of chunks.
        """
        return self.chunker.chunk(documents=[document], **kwargs)

    def _clean_resource(self):
        """
        Clean the retriever resource.
        """
        self.vector_database.clean_resource()
        self.bm25.clean_resource()

    def _build_gen_context_input(
        self, chunks: list[Document], whole_document: Document
    ) -> list[tuple[Document, Document]]:
        """
        Build the input for generating contextual chunks.

        Args:
            chunks (list[Document]): The list of chunks.
            whole_document (Document): The whole document.

        Returns:
            list[tuple[Document, Document]]: A list of tuples containing the chunk and the whole document.
        """
        return [(chunk, whole_document) for chunk in chunks]

    def _ingest_db(
        self,
        file_or_folder_paths: list[str],
        batch_embedding: int = 100,
        num_workers: int = 4,
    ) -> None:
        """
        Ingest documents into the Contextual RAG database.

        Args:
            file_paths (list[str]): List of file paths to be ingested.
            batch_embedding (int): Batch size for embedding documents.
            num_workers (int): Number of workers for parallel processing.
        """
        documents = self.processor.load_data(file_or_folder_paths=file_or_folder_paths)

        chunks = []
        for doc in tqdm(
            documents, desc="Splitting documents into chunks ...", unit="doc"
        ):
            chunks.append(self._split_document(doc, show_progress=False))

        gen_contextual_inputs: list[tuple[Document, Document]] = []
        for document, chunk in zip(documents, chunks):
            gen_contextual_inputs.extend(self._build_gen_context_input(chunk, document))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    gen_contextual_chunk,
                    chunk,
                    whole_document,
                    self.llm,
                    self.gen_contextual_prompt_template,
                )
                for chunk, whole_document in gen_contextual_inputs
            ]

            contextual_chunks = []
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Generating contextual chunks (num_workers={num_workers})",
            ):
                contextual_chunks.append(future.result())

        embeddings = self.embed_model.get_batch_document_embedding(
            documents=contextual_chunks, batch_size=batch_embedding
        )

        embeded_chunks = [
            RetrieverIngestInput(
                id=doc.id,
                document=doc.document,
                embedding=e.embedding,
                metadata=doc.metadata,
            )
            for doc, e in zip(contextual_chunks, embeddings)
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
    ) -> list[Document]:
        """
        Retrieve documents from the database.

        Args:
            query (RetrieverQueryType): The query to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments for the retrieval process.

        Returns:
            list[Document]: A list of retrieved documents.
        """
        semantic_result = self.semantic_retriever.retrieve(query=query, top_k=top_k)
        bm25_result = self.bm25_retriever.retrieve(query=query, top_k=top_k)
        return semantic_result, bm25_result

    def _search(
        self,
        *,
        query: RetrieverQueryType,
        top_k: int = 5,
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

        retrieved_time, (semantic_results, bm25_results) = run_fuction_return_time(
            self._retrieve_db,
            query=query,
            top_k=top_k,
            **kwargs,
        )

        reranked_results = self.reranker.rerank(
            query=query,
            results=[semantic_results, bm25_results],
            top_k=top_k,
        )

        contexts = "\n ============ \n".join(
            result.document for result in reranked_results
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
            retrieved_results=reranked_results,
            retrieved_time=retrieved_time,
            generation_time=generation_time,
        )
