from tqdm import tqdm
from loguru import logger

from rag_colls.types.reranker import RerankerResult
from rag_colls.core.base.rerankers.base import BaseReranker
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class ChainReranker(BaseReranker):
    """
    ChainReranker is a reranker that uses a chain of rerankers to rerank the results.
    """

    def __init__(
        self,
        *,
        rerankers: list[BaseReranker],
        top_ks: list[int],
        show_progress: bool = False,
    ):
        """
        Initialize the ChainReranker class.

        Args:
            rerankers (list[BaseReranker]): The list of rerankers to use in the chain.
            top_ks: (list[int]): The list of top_ks to use for each reranker in the chain. Must be in descending order.
            show_progress (bool): Whether to show progress during reranking.
        """
        self._check_valid_rerankers(rerankers=rerankers)
        self.rerankers = rerankers

        self._check_valid_top_ks(top_ks=top_ks)
        self.top_ks = top_ks

        self.show_progress = show_progress

        logger.info(f"Initialized ChainReranker with {len(rerankers)} rerankers.")

    def __str__(self):
        return f"ChainReranker(rerankers={self.rerankers})"

    def _check_valid_top_ks(self, top_ks: list[int]) -> bool:
        """
        Check if the provided top_ks are valid.

        Args:
            top_ks (list[int]): The list of top_ks to check.

        Returns:
            bool: True if the top_ks are valid, False otherwise.
        """
        # Check if all top_ks are descending
        assert all(top_ks[i] >= top_ks[i + 1] for i in range(len(top_ks) - 1)), (
            "top_ks must be in descending order"
        )

        assert len(top_ks) == len(self.rerankers), (
            "top_ks must be the same length as rerankers"
        )

    def _check_valid_rerankers(self, rerankers: list[BaseReranker]) -> bool:
        """
        Check if the provided rerankers are valid.

        Args:
            rerankers (list[BaseReranker]): The list of rerankers to check.

        Returns:
            bool: True if the rerankers are valid, False otherwise.
        """
        for reranker in rerankers:
            assert isinstance(reranker, BaseReranker), f"Invalid reranker: {reranker}"

    def __repr__(self):
        return self.__str__()

    def _rerank(
        self,
        query: RetrieverQueryType,
        results: list[list[RetrieverResult]] | list[RetrieverResult],
        **kwargs,
    ) -> list[RerankerResult]:
        """
        Rerank the results using the chain of rerankers.

        Args:
            query (RetrieverQueryType): The query to rerank the results for.
            results (list[list[RetrieverResult]] | list[RetrieverResult]): The results to rerank. Can be a list of retriever results or a list of lists of retriever results.
            top_k (int): The `MAXIMUM` number of top results to return.
            **kwargs: Additional arguments for the reranker.

        Returns:
            list[RerankerResult]: The reranked results.
        """
        del kwargs["top_k"]  # Remove top_k from kwargs

        bar = tqdm(
            zip(self.rerankers, self.top_ks),
            total=len(self.rerankers),
            desc="Reranking ...",
            disable=not self.show_progress,
        )

        for reranker, k in bar:
            results = reranker.rerank(query=query, results=results, top_k=k, **kwargs)

        return results
