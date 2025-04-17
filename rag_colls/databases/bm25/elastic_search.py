import elasticsearch
from elasticsearch.helpers import bulk
from loguru import logger

from rag_colls.core.base.database.bm25 import BaseBM25RetrieverProvider
from rag_colls.types.retriever import (
    RetrieverResult,
    RetrieverQueryType,
    RetrieverIngestInput,
)


class ElasticSearch(BaseBM25RetrieverProvider):
    """
    Wrapper for the Elasticsearch library: `https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html`
    """

    def __init__(self, host: str | None = None, **kwargs):
        if host is None:
            raise ValueError("Host must be provided")

        self.index_name = kwargs.get("index_name", "documents_bm25")
        self.es = elasticsearch.Elasticsearch(host)

        if not self._test_connection():
            raise ValueError("Failed to connect to Elasticsearch")

        logger.info(f"Connected to Elasticsearch at {host}")

        # Create index if it doesn't exist
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
                index=self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "metadata": {"type": "object"},
                        }
                    }
                },
            )

    def _test_connection(self):
        """
        Test the connection to the Elasticsearch database.
        """
        try:
            return self.es.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False

    def _clean_resource(self):
        """
        Clean the Elasticsearch resource.
        """
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            logger.debug(f"Cleaned up Elasticsearch index {self.index_name}")
        else:
            logger.debug(f"Elasticsearch index {self.index_name} does not exist.")

    def _build_corpus(self, documents: list[RetrieverIngestInput]):
        """
        Build the corpus for Elasticsearch indexing.
        """
        return [
            {
                "_index": self.index_name,
                "_source": {
                    "text": doc.document,
                    "metadata": {"id": doc.id, **doc.metadata},
                },
            }
            for doc in documents
        ]

    def _index_documents(self, documents: list[RetrieverIngestInput], **kwargs):
        """
        Ingest documents into the Elasticsearch database.

        Args:
            documents (list[Document]): List of documents to ingest.
        """
        corpus = self._build_corpus(documents)

        bulk_size = kwargs.get("bulk_size", 5)

        # Index the documents
        success, failed = bulk(
            self.es,
            corpus,
            chunk_size=bulk_size,
            request_timeout=len(corpus),
            raise_on_error=False,
        )

        logger.info(f"Indexed {success} documents")
        logger.info(f"Failed to index {failed} documents")

        if failed:
            logger.warning(f"Failed to index {len(failed)} documents")

        # Refresh the index to make the documents searchable
        self.es.indices.refresh(index=self.index_name)

    def _add_documents(self, documents: list[RetrieverIngestInput], **kwargs):
        """
        Ingest documents into the Elasticsearch database.

        Args:
            documents (list[Document]): List of documents to ingest.
        """
        self._index_documents(documents, **kwargs)

    def _search(
        self, query: RetrieverQueryType, top_k: int = 5, **kwargs
    ) -> list[RetrieverResult]:
        """
        Perform a search using the Elasticsearch retriever.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to return.

        Returns:
            list[RetrieverResult]: List of retrieved documents with their metadata.
        """
        assert isinstance(query, str), "Query must be a string."

        # Perform the search
        response = self.es.search(
            index=self.index_name,
            body={"query": {"match": {"text": query}}, "size": top_k},
        )

        search_results: list[RetrieverResult] = []

        # Process the results
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            search_results.append(
                RetrieverResult(
                    id=doc["metadata"]["id"],
                    score=hit["_score"],
                    document=doc["text"],
                    metadata=doc["metadata"],
                )
            )

        # Normalize scores to [0, 1]
        if search_results:
            max_score = max(result.score for result in search_results)
            min_score = min(result.score for result in search_results)

            for result in search_results:
                result.score = (
                    (result.score - min_score) / (max_score - min_score)
                    if max_score != min_score
                    else 0.0
                )

        return search_results
