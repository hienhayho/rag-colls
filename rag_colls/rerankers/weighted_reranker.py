from rag_colls.core.base.rerankers.base import BaseReranker

from rag_colls.types.reranker import RerankerResult
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class WeightedReranker(BaseReranker):
    """
    Weighted Reranker that assigns weights to the results based on their relevance score.
    """

    def __init__(self, weights: list[float]):
        """
        Initialize the WeightedReranker with weights.

        Args:
            weights (list[float]): List of weights for each result.
        """
        self.weights = weights

    def _rerank(
        self,
        query: RetrieverQueryType,
        results: list[list[RetrieverResult]],
        top_k: int = 10,
    ) -> list[RerankerResult]:
        """
        Rerank the results based on the query and weights.

        Args:
            query (RetrieverQueryType): The query to rerank the results for.
            results (list[list[RetrieverResult]]): The results to rerank.
            top_k (int): The number of top results to return.

        Returns:
            list[RerankerResult]: The reranked results.
        """
        reranked_results: list[RerankerResult] = []

        assert len(results) == len(self.weights), (
            "Number of results must match number of weights."
        )

        # Make sure weights are normalized to sum to 1
        total_weight = sum(self.weights)

        assert total_weight > 0, "Weights must sum to a positive value."

        if total_weight != 1:
            self.weights = [w / total_weight for w in self.weights]

        for i, result_set in enumerate(results):
            weighted_results = [
                RerankerResult(
                    id=result.id,
                    document=result.document,
                    score=result.score * self.weights[i],
                    metadata=result.metadata,
                )
                for result in result_set
            ]

            reranked_results.extend(weighted_results)

        reranked_results.sort(key=lambda x: x.score, reverse=True)

        return reranked_results[:top_k]
