from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore, TextNode

from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.core.base.rerankers.base import BaseReranker
from rag_colls.llms.integrations.llama_index import LlamaIndexLLM
from rag_colls.types.reranker import RerankerResult
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class LLMReranker(BaseReranker):
    def __init__(self, llm: BaseCompletionLLM):
        """
        Initialize the LLMReranker with a language model and top_n.

        Args:
            llm (BaseCompletionLLM): The language model to use for reranking.
            top_n (int): The number of top results to return.
        """
        self.llm = LlamaIndexLLM(llm=llm)

    def __str__(self):
        return f"LLMReranker(llm={self.llm})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_llm(
        cls,
        llm: BaseCompletionLLM,
    ) -> BaseReranker:
        """
        Create an instance of LLMReranker.

        Args:
            llm (BaseCompletionLLM): The language model to use for reranking.

        Returns:
            BaseReranker: An instance of LLMReranker.
        """
        return cls(llm=llm)

    def is_support_aggregate_results(self):
        """
        Check if the reranker supports aggregating results.

        Returns:
            bool: True if the reranker supports aggregating results, False otherwise.
        """
        return True

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
            results (list[list[RetrieverResult]]): The results to rerank.
            top_k (int): The `MAXIMUM` number of results to return.
            **kwargs: Additional arguments for the reranker.

        Returns:
            list[RetrieverResult]: The reranked results.
        """
        reranker = LLMRerank(top_n=top_k, llm=self.llm)
        # Flatten the results if they are nested
        if isinstance(results[0], list):
            flattened_results = [item for sublist in results for item in sublist]
        else:
            flattened_results = results

        nodes = [
            NodeWithScore(
                node=TextNode(
                    id_=result.id,
                    text=result.document,
                    metadata=result.metadata,
                ),
                score=result.score,
            )
            for result in flattened_results
        ]

        reranked_nodes = reranker.postprocess_nodes(nodes, query_str=query)

        if len(reranked_nodes) == 0:
            return []

        # Normalize scores to be between 0 and 1
        max_score = max(node.score for node in reranked_nodes)
        min_score = min(node.score for node in reranked_nodes)
        score_range = max_score - min_score
        if score_range > 0:
            for node in reranked_nodes:
                node.score = (node.score - min_score) / score_range
        else:
            for node in reranked_nodes:
                node.score = 0.0

        reranked_results = [
            RetrieverResult(
                id=node.node.node_id,
                document=node.node.text,
                score=node.score,
                metadata=node.node.metadata,
            )
            for node in reranked_nodes
        ]

        # Simply return the reranked results since the reranker already handles the top_k
        return reranked_results
