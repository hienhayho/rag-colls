import asyncio
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document as LlamaIndexDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser

from rag_colls.types.core.document import Document
from rag_colls.core.base.chunkers.base import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that chunks documents based on semantic similarity.
    """

    supported_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]

    defaul_model = "text-embedding-ada-002"

    def __init__(
        self,
        embed_model_name: str | None = None,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
    ):
        if not embed_model_name:
            embed_model_name = self.defaul_model

        assert embed_model_name in self.supported_models, (
            f"Model {embed_model_name} is not supported. Please use openai embedding models."
        )

        self.embed_model_name = embed_model_name
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            embed_model=OpenAIEmbedding(model=embed_model_name),
        )

    def _chunk(self, documents: list[Document], show_progress: bool = True, **kwargs):
        """
        Chunk the documents based on semantic similarity.

        Args:
            documents (list[Document]): List of documents to be chunked.
            **kwargs: Additional keyword arguments for the chunking function.

        Returns:
            list[Document]: List of chunked documents.
        """
        preprocessed_documents = [
            LlamaIndexDocument(
                doc_id=doc.id,
                text=doc.document,
                metadata=doc.metadata,
            )
            for doc in documents
        ]

        nodes = self.node_parser.get_nodes_from_documents(
            documents=preprocessed_documents, show_progress=show_progress
        )

        return [Document(document=node.text, metadata=node.metadata) for node in nodes]

    async def _achunk(
        self, documents: list[Document], show_progress: bool = False, **kwargs
    ):
        """
        Asynchronously chunk the documents based on semantic similarity.

        Args:
            documents (list[Document]): List of documents to be chunked.
            **kwargs: Additional keyword arguments for the chunking function.

        Returns:
            list[Document]: List of chunked documents.
        """
        # For now, we will just call the synchronous method
        return await asyncio.to_thread(self._chunk, documents, **kwargs)
