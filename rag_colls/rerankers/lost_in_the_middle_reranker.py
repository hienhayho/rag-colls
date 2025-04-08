# Ref: https://github.com/deepset-ai/haystack/blob/main/haystack/components/rankers/lost_in_the_middle.py
from loguru import logger

from rag_colls.types.reranker import RerankerResult
from rag_colls.core.base.rerankers.base import BaseReranker
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult


class LostInTheMiddleReranker(BaseReranker):
    """
    Ranks documents based on the 'lost in the middle' order so that the most relevant documents are either at the
    beginning or end, while the least relevant are in the middle.

    LostInTheMiddleRanker assumes that some prior component in the pipeline has already ranked documents by relevance
    and requires no query as input but only documents. It is typically used as the last component before building a
    prompt for an LLM to prepare the input context for the LLM.

    Lost in the Middle ranking lays out document contents into LLM context so that the most relevant contents are at
    the beginning or end of the input context, while the least relevant is in the middle of the context. See the
    paper ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) for more
    details.

    **Example:**
    ```python
    Before reranking:
    ranked_docs_idx = [4, 6, 1, 2, 3, 5, 7]

    After reranking:
    ranked_docs_idx = [4, 1, 3, 7, 5, 2, 6]
    ```
    """

    def __init__(self, word_count_threshold: int | None = None):
        """
        Initialize the LostInTheMiddleReranker class.

        Args:
            word_count_threshold (int | None): The threshold for the maximum number of words in the reranked documents.
                If None, no threshold is applied.
        """
        self.word_count_threshold = word_count_threshold

        logger.warning(
            "You should only use LostInTheMiddleReranker when the documents are already ranked by relevance."
        )
        logger.info(
            f"LostInTheMiddleReranker with word_count_threshold={self.word_count_threshold}"
        )

    def __str__(self):
        return (
            f"LostInTheMiddleReranker(word_count_threshold={self.word_count_threshold})"
        )

    def __repr__(self):
        return self.__str__()

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
            list[RerankerResult]: The reranked results.
        """
        assert isinstance(results[0], RetrieverResult), (
            "LostInTheMiddleReranker only supports a single list of results."
        )

        documents_to_reorder = results[:top_k]
        if len(documents_to_reorder) == 1:
            return documents_to_reorder

        word_count = 0
        document_index = list(range(len(documents_to_reorder)))
        lost_in_the_middle_indices = [0]

        # If word count threshold is set and the first document has content, calculate word count for the first document
        if self.word_count_threshold and documents_to_reorder[0].document:
            word_count = len(documents_to_reorder[0].document.split())

            # If the first document already meets the word count threshold, return it
            if word_count >= self.word_count_threshold:
                logger.warning(
                    "The first document already meets the word count threshold. "
                    "Returning only the first document."
                )
                return [
                    RerankerResult(
                        id=documents_to_reorder[0].id,
                        score=documents_to_reorder[0].score,
                        document=documents_to_reorder[0].document,
                        metadata=documents_to_reorder[0].metadata,
                    )
                ]

        # Start from the second document and create "lost in the middle" order
        for doc_idx in document_index[1:]:
            # Calculate the index at which the current document should be inserted
            insertion_index = (
                len(lost_in_the_middle_indices) // 2
                + len(lost_in_the_middle_indices) % 2
            )

            # Insert the document index at the calculated position
            lost_in_the_middle_indices.insert(insertion_index, doc_idx)

            # If word count threshold is set and the document has content, calculate the total word count
            if self.word_count_threshold and documents_to_reorder[doc_idx].document:
                word_count += len(documents_to_reorder[doc_idx].document.split())  # type: ignore[union-attr]

                # If the total word count meets the threshold, stop processing further documents
                if word_count >= self.word_count_threshold:
                    break

        # Documents in the "lost in the middle" order
        ranked_docs = [documents_to_reorder[idx] for idx in lost_in_the_middle_indices]

        return [
            RerankerResult(
                id=doc.id,
                score=doc.score,
                document=doc.document,
                metadata=doc.metadata,
            )
            for doc in ranked_docs
        ]
