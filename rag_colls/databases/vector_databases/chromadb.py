import asyncio
import chromadb
from loguru import logger

from rag_colls.types.retriever import RetrieverIngestInput, RetrieverResult
from rag_colls.core.base.database.vector_database import BaseVectorDatabase


class ChromaVectorDatabase(BaseVectorDatabase):
    def __init__(self, persistent_directory: str, collection_name: str):
        """
        Initialize the Chroma vector database.

        Args:
            persistent_directory (str): Directory to persist the database.
            collection_name (str): Name of the collection to create.
        """
        self.persistent_directory = persistent_directory
        self.client = chromadb.PersistentClient(
            path=persistent_directory,
        )
        self._test_connection()

        self.collection_name = collection_name

        self._create_collection(collection_name)

        logger.success("ChromaVectorDatabase initialized successfully !!!")

    def __str__(self):
        return f"ChromaVectorDatabase(persistent_directory={self.persistent_directory}, collection_name={self.collection_name})"

    def _test_connection(self):
        """
        Test the connection to the Chroma vector database.
        """
        self.client.heartbeat()

    def _clean_resource(self):
        """
        Clean the Chroma vector database resource.
        """
        self.client.delete_collection(self.collection_name)
        logger.debug(f"Cleaned up Chroma vector database at {self.collection_name}")

    def _check_collection_exists(self, collection_name):
        """
        Check if a collection exists in the Chroma vector database.

        Args:
            collection_name (str): Name of the collection to check.
        """
        try:
            self.client.get_collection(name=collection_name)
        except Exception:
            return False
        return True

    def _create_collection(self, collection_name, **kwargs):
        """
        Create a collection in the Chroma vector database.

        Args:
            collection_name (str): Name of the collection to create.
            **kwargs: Additional arguments for the collection.
        """
        if self._check_collection_exists(collection_name):
            return

        self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.success(f"Collection {collection_name} created successfully !!!")

    def _add_documents(self, documents: list[RetrieverIngestInput], **kwargs):
        """
        Add documents to the Chroma vector database.

        Args:
            documents (list[str]): List of documents to add.
            **kwargs: Additional arguments for the add operation.
        """
        if not self._check_collection_exists(self.collection_name):
            logger.error(f"Collection {self.collection_name} does not exist.")
            return

        collection = self.client.get_collection(self.collection_name)

        metadatas = [{"document": doc.document, **doc.metadata} for doc in documents]
        embeddings = [doc.embedding for doc in documents]

        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[doc.id for doc in documents],
        )

        logger.debug(f"Count: {collection.count()}")

        logger.success(f"Added: {len(documents)} documents.")

    def _delete_documents(self, document_ids: list[str], **kwargs):
        """
        Delete documents from the Chroma vector database.

        Args:
            document_ids (list[str]): List of document IDs to delete.
            **kwargs: Additional arguments for the delete operation.
        """
        if not self._check_collection_exists(self.collection_name):
            logger.error(f"Collection {self.collection_name} does not exist.")
            return

        collection = self.client.get_collection(self.collection_name)
        collection.delete(ids=document_ids)

        logger.success(f"Deleted: {len(document_ids)} documents.")

    def _search(
        self, query_embedding: list[float], top_k: int = 5, **kwargs
    ) -> list[RetrieverResult]:
        """
        Search for documents in the Chroma vector database.

        Args:
            query_embedding (list[float]): The embedding of the query.
            top_k (int): Number of top results to return.
            **kwargs: Additional arguments for the search operation.

        Returns:
            list[RetrieverResult]: List of retrieved documents with their metadata.
        """
        if not self._check_collection_exists(self.collection_name):
            logger.error(f"Collection {self.collection_name} does not exist.")
            return []

        collection = self.client.get_collection(self.collection_name)
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]

        # We use cosine_similarity to have larger score for more similar documents
        scores = [1 - s for s in results["distances"][0]]

        documents = [
            RetrieverResult(
                id=doc_id,
                score=score,
                document=metadata["document"],
                metadata=metadata,
            )
            for doc_id, metadata, score in zip(ids, metadatas, scores)
        ]

        return documents

    async def _asearch(
        self, query_embedding: list[float], top_k: int = 5, **kwargs
    ) -> list[RetrieverResult]:
        """
        Asynchronous search for documents in the Chroma vector database.

        Args:
            query_embedding (list[float]): The embedding of the query.
            top_k (int): Number of top results to return.
            **kwargs: Additional arguments for the search operation.

        Returns:
            list[RetrieverResult]: List of retrieved documents with their metadata.
        """
        return await asyncio.to_thread(
            self._search,
            query_embedding=query_embedding,
            top_k=top_k,
            **kwargs,
        )
