from abc import ABC, abstractmethod
from rag_colls.types.reranker import RerankerResult
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class BaseReranker(ABC):
    @abstractmethod
    def _rerank(
        self,
        query: RetrieverQueryType,
        results: list[list[RetrieverResult]],
        top_k: int = 10,
        **kwargs,
    ) -> list[RerankerResult]:
        """
        Rerank the results based on the query.

        Args:
            query (RetrieverQueryType): The query to rerank the results for.
            results (list[list[RetrieverResult]]): The results to rerank.
            top_k (int): The `MAXIMUM` number of top results to return.
            **kwargs: Additional arguments for the reranker.

        Returns:
            list[RetrieverResult]: The reranked results.
        """
        raise NotImplementedError("Rerank method not implemented.")

    def rerank(
        self,
        query: RetrieverQueryType,
        results: list[list[RetrieverResult]],
        top_k: int = 10,
        **kwargs,
    ) -> list[RerankerResult]:
        """
        Rerank the results based on the query.

        Args:
            query (RetrieverQueryType): The query to rerank the results for.
            results (list[list[RetrieverResult]]): The results to rerank.
            top_k (int): The `MAXIMUM` number of top results to return.
            **kwargs: Additional arguments for the reranker.

        Returns:
            list[RetrieverResult]: The reranked results.
        """
        return self._rerank(query=query, results=results, top_k=top_k, **kwargs)
