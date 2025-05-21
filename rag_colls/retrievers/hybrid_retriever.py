from rag_colls.core.base.retrievers.base import BaseRetriever
from rag_colls.core.base.rerankers.base import BaseReranker
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines multiple retrievers.
    """

    def __init__(self, retrievers: list[BaseRetriever], reranker: BaseReranker):
        """
        Initialize the hybrid retriever with a list of retrievers.

        Args:
            retrievers (list[BaseRetriever]): List of retrievers to combine.
            reranker (BaseReranker): Reranker instance to use for re-ranking.
        """

        self.retrievers = retrievers
        self.reranker = reranker

    def _retrieve(self, query: RetrieverQueryType, **kwargs) -> list[RetrieverResult]:
        """
        Retrieve documents using the combined retrievers and re-rank them.
        Args:
            query (RetrieverQueryType): The query to retrieve documents for.
            **kwargs: Additional arguments to pass to the retrievers.
        Returns:
            list[RetrieverResult]: List of retrieved and re-ranked documents.
        """

        combined_results = []
        for retriever in self.retrievers:
            results = retriever.retrieve(query, **kwargs)
            combined_results.append(results)

        reranked_results = self.reranker.rerank(query, combined_results, **kwargs)

        return reranked_results

    def _clean_resource(self):
        """
        Clean up resources used by the retriever.
        """

        for retriever in self.retrievers:
            retriever.clean_resource()
