from abc import ABC, abstractmethod
from rag_colls.types.reranker import RerankerResult
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class BaseReranker(ABC):
    @abstractmethod
    def __str__(self):
        """
        Get the string representation of the reranker.

        Returns:
            str: String representation of the reranker.
        """
        raise NotImplementedError("String representation not implemented.")

    @abstractmethod
    def _rerank(
        self,
        query: RetrieverQueryType,
        results: list[list[RetrieverResult]] | list[RetrieverResult],
        top_k: int = 10,
        **kwargs,
    ) -> list[RerankerResult]:
        """
        Rerank the results based on the query.

        Args:
            query (RetrieverQueryType): The query to rerank the results for.
            results (list[list[RetrieverResult]] | list[RetrieverResult]): The results to rerank. Can be a list of retriever results or a list of lists of retriever results.
            top_k (int): The `MAXIMUM` number of top results to return.
            **kwargs: Additional arguments for the reranker.

        Returns:
            list[RetrieverResult]: The reranked results.
        """
        raise NotImplementedError("Rerank method not implemented.")

    def is_support_aggregate_results(self) -> bool:
        """
        **NOTE**: This method is default `False` and should be overridden by the support aggregating rerankers.

        Check if the reranker supports aggregating results.

        Returns:
            bool: True if the reranker supports aggregating results, False otherwise.
        """
        return False

    def rerank(
        self,
        query: RetrieverQueryType,
        results: list[list[RetrieverResult]] | list[RetrieverResult],
        top_k: int = 10,
        **kwargs,
    ) -> list[RerankerResult]:
        """
        Rerank the results based on the query.

        Args:
            query (RetrieverQueryType): The query to rerank the results for.
            results (list[list[RetrieverResult]] | list[RetrieverResult]): The results to rerank. Can be a list of retriever results or a list of lists of retriever results.
            top_k (int): The `MAXIMUM` number of top results to return.
            **kwargs: Additional arguments for the reranker.

        Returns:
            list[RetrieverResult]: The reranked results.
        """
        rerank_results = self._rerank(query=query, results=results, **kwargs)

        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for result in rerank_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)

        return unique_results[:top_k]
