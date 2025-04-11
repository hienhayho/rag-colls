from rag_colls.core.base.embeddings.base import BaseEmbedding
from rag_colls.core.base.retrievers.base import BaseRetriever
from rag_colls.core.base.database.vector_database import BaseVectorDatabase

from rag_colls.core.settings import GlobalSettings
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class VectorDatabaseRetriever(BaseRetriever):
    """
    A retriever that uses vector databases to retrieve documents based on a query.
    """

    def __init__(
        self, vector_db: BaseVectorDatabase, embed_model: BaseEmbedding | None = None
    ):
        """
        Initialize the VectorDatabaseRetriever class.

        Args:
            vector_db: The vector database to use for retrieval.
            embed_model: The embedding model to use for generating embeddings.
        """
        self.vector_db = vector_db
        self.embed_model = embed_model or GlobalSettings.embed_model

    @classmethod
    def from_vector_db(
        cls, vector_db: BaseVectorDatabase, embed_model: BaseEmbedding | None = None
    ) -> "VectorDatabaseRetriever":
        """
        Create an instance of VectorDatabaseRetriever from a vector database.

        Args:
            vector_db: The vector database to use for retrieval.

        Returns:
            VectorDatabaseRetriever: An instance of the retriever.
        """
        return cls(vector_db=vector_db, embed_model=embed_model)

    def _retrieve(self, query: RetrieverQueryType, **kwargs) -> list[RetrieverResult]:
        """
        Retrieve documents based on the query.

        Args:
            query: The query to search for.
            **kwargs: Additional keyword arguments for the retrieval process.

        Returns:
            list[RetrieverResult]: A list of retrieved documents.
        """
        query_embedding = query
        if isinstance(query, str):
            query_embedding = self.embed_model.get_query_embedding(query).embedding

        return self.vector_db.search(query_embedding=query_embedding, **kwargs)

    def _clean_resource(self):
        """
        Clean the retriever resource.
        """
        self.vector_db.clean_resource()
