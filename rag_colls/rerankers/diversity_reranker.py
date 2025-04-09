# Ref: https://github.com/deepset-ai/haystack/blob/45cd6f43d6651308b93be4f9c7aa6be15079272c/haystack/components/rankers/sentence_transformers_diversity.py
from loguru import logger
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from rag_colls.types.reranker import RerankerResult
from rag_colls.core.base.rerankers.base import BaseReranker
from rag_colls.core.serialization import (
    default_to_dict,
    default_from_dict,
    serialize_hf_model_kwargs,
    deserialize_hf_model_kwargs,
)
from rag_colls.types.retriever import RetrieverQueryType, RetrieverResult

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "The 'sentence-transformers' package is not installed. Please install it using 'pip install \"sentence-transformers>=3.0.0\"'."
    )


class DiversityRankingStrategy(Enum):
    """
    The strategy to use for diversity ranking.
    """

    GREEDY_DIVERSITY_ORDER = "greedy_diversity_order"
    MAXIMUM_MARGIN_RELEVANCE = "maximum_margin_relevance"

    def __str__(self) -> str:
        """
        Convert a Strategy enum to a string.
        """
        return self.value

    @staticmethod
    def from_str(string: str) -> "DiversityRankingStrategy":
        """
        Convert a string to a Strategy enum.
        """
        enum_map = {e.value: e for e in DiversityRankingStrategy}
        strategy = enum_map.get(string)
        if strategy is None:
            msg = f"Unknown strategy '{string}'. Supported strategies are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return strategy


class DiversityRankingSimilarity(Enum):
    """
    The similarity metric to use for comparing embeddings.
    """

    DOT_PRODUCT = "dot_product"
    COSINE = "cosine"

    def __str__(self) -> str:
        """
        Convert a Similarity enum to a string.
        """
        return self.value

    @staticmethod
    def from_str(string: str) -> "DiversityRankingSimilarity":
        """
        Convert a string to a Similarity enum.
        """
        enum_map = {e.value: e for e in DiversityRankingSimilarity}
        similarity = enum_map.get(string)
        if similarity is None:
            msg = f"Unknown similarity metric '{string}'. Supported metrics are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return similarity


class SentenceTransformersDiversityRanker(BaseReranker):
    """
    A Diversity Ranker based on Sentence Transformers that implements two strategies for reranking results.

    This ranker uses sentence transformers to compute embeddings and then applies one of two strategies:

    1. Maximum Margin Relevance (MMR):
        - Balances relevance and diversity using a lambda threshold
        - A higher lambda (closer to 1) favors relevance
        - A lower lambda (closer to 0) favors diversity
        - Based on the paper: "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries"

    2. Greedy Diversity Order:
        - Maximizes diversity while maintaining relevance
        - Starts with the most relevant document
        - Iteratively selects documents that are least similar to already selected ones
        - Uses sentence transformer embeddings for similarity computation

    The ranker supports both cosine and dot product similarity metrics and can be configured with various
    model parameters and preprocessing options.

    ### Usage example
    ```python
    from rag_colls.types.retriever import RetrieverResult
    from rag_colls.rerankers import SentenceTransformersDiversityRanker

    ranker = SentenceTransformersDiversityRanker(
        model="sentence-transformers/all-MiniLM-L6-v2",
        strategy="maximum_margin_relevance",
        lambda_threshold=0.7
    )

    results = [
        RetrieverResult(
            id="1",
            score=0.9,
            document="First document content",
            metadata={}
        ),
        RetrieverResult(
            id="2",
            score=0.8,
            document="Second document content",
            metadata={}
        )
    ]

    reranked_results = ranker._rerank(
        query="search query",
        results=results,
        top_k=5
    )
    ```
    """

    def __init__(  # noqa: PLR0913 # pylint: disable=too-many-positional-arguments
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 10,
        device: str = "auto",
        token: Optional[str] = None,
        similarity: Union[str, DiversityRankingSimilarity] = "cosine",
        query_prefix: str = "",
        query_suffix: str = "",
        document_prefix: str = "",
        document_suffix: str = "",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        strategy: Union[str, DiversityRankingStrategy] = "greedy_diversity_order",
        lambda_threshold: float = 0.5,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ):
        """
        Initialize the SentenceTransformersDiversityRanker.

        Args:
            model (str): Name or path of the sentence transformer model to use.
            top_k (int): Maximum number of results to return.
            device (str): Device to run the model on ("cpu", "cuda", or "auto").
            token (Optional[str]): HuggingFace API token for private models.
            similarity (Union[str, DiversityRankingSimilarity]): Similarity metric to use ("cosine" or "dot_product").
            query_prefix (str): Prefix to add to queries before embedding.
            query_suffix (str): Suffix to add to queries before embedding.
            document_prefix (str): Prefix to add to documents before embedding.
            document_suffix (str): Suffix to add to documents before embedding.
            meta_fields_to_embed (Optional[List[str]]): List of metadata fields to include in embeddings.
            embedding_separator (str): Separator to use when concatenating metadata fields.
            strategy (Union[str, DiversityRankingStrategy]): Reranking strategy to use ("maximum_margin_relevance" or "greedy_diversity_order").
            lambda_threshold (float): Trade-off parameter between relevance and diversity (0.0 to 1.0).
            model_kwargs (Optional[Dict[str, Any]]): Additional arguments for model initialization.
            tokenizer_kwargs (Optional[Dict[str, Any]]): Additional arguments for tokenizer initialization.
            config_kwargs (Optional[Dict[str, Any]]): Additional arguments for model configuration.
            backend (Literal["torch", "onnx", "openvino"]): Backend to use for model inference.

        Raises:
            ValueError: If top_k is not positive or lambda_threshold is not between 0 and 1.
        """

        self.model_name_or_path = model
        if top_k is None or top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.top_k = top_k
        self.device = (
            torch.device(device)
            if device != "auto"
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.token = token
        self.model = None
        self.similarity = (
            DiversityRankingSimilarity.from_str(similarity)
            if isinstance(similarity, str)
            else similarity
        )
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.query_suffix = query_suffix
        self.document_suffix = document_suffix
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.strategy = (
            DiversityRankingStrategy.from_str(strategy)
            if isinstance(strategy, str)
            else strategy
        )
        self.lambda_threshold = lambda_threshold or 0.5
        self._check_lambda_threshold(self.lambda_threshold, self.strategy)
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
        self.backend = backend
        if self.model is None:
            logger.info(
                f"Initializing SentenceTransformersDiversityRanker with model '{self.model_name_or_path}' on device '{self.device}'"
            )
            logger.info(
                f"Using strategy '{self.strategy}' with lambda threshold '{self.lambda_threshold}'"
            )
            self.model = SentenceTransformer(
                model_name_or_path=self.model_name_or_path,
                device=self.device,
                token=self.token.resolve_value() if self.token else None,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                config_kwargs=self.config_kwargs,
                backend=self.backend,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            model=self.model_name_or_path,
            top_k=self.top_k,
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            similarity=str(self.similarity),
            query_prefix=self.query_prefix,
            query_suffix=self.query_suffix,
            document_prefix=self.document_prefix,
            document_suffix=self.document_suffix,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            strategy=str(self.strategy),
            lambda_threshold=self.lambda_threshold,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            backend=self.backend,
        )
        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(
                serialization_dict["init_parameters"]["model_kwargs"]
            )
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersDiversityRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[RetrieverResult]) -> List[str]:
        """
        Prepare texts for embedding by combining document content with selected metadata fields.

        Args:
            documents (List[RetrieverResult]): List of documents to prepare for embedding.

        Returns:
            List[str]: List of prepared texts ready for embedding, where each text is constructed by:
                - Adding document_prefix
                - Concatenating selected metadata fields (if any) with embedding_separator
                - Adding the document content
                - Adding document_suffix
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.metadata[key])
                for key in self.meta_fields_to_embed
                if key in doc.metadata and doc.metadata[key]
            ]
            text_to_embed = (
                self.document_prefix
                + self.embedding_separator.join(
                    meta_values_to_embed + [doc.document or ""]
                )
                + self.document_suffix
            )
            texts_to_embed.append(text_to_embed)

        return texts_to_embed

    def _greedy_diversity_order(
        self, query: str, documents: List[RetrieverResult]
    ) -> List[RetrieverResult]:
        """
        Rerank documents using the Greedy Diversity Order strategy.

        This strategy:
        1. Starts with the document most similar to the query
        2. Iteratively selects the document that is least similar to all previously selected documents
        3. Uses sentence transformer embeddings to compute similarities

        Args:
            query (str): The search query.
            documents (List[RetrieverResult]): List of documents to rerank.

        Returns:
            List[RetrieverResult]: Reranked documents ordered to maximize diversity while maintaining relevance.
        """
        texts_to_embed = self._prepare_texts_to_embed(documents)

        doc_embeddings, query_embedding = self._embed_and_normalize(
            query, texts_to_embed
        )

        n = len(documents)
        selected: List[int] = []

        # Compute the similarity vector between the query and documents
        query_doc_sim = query_embedding @ doc_embeddings.T

        # Start with the document with the highest similarity to the query
        selected.append(int(torch.argmax(query_doc_sim).item()))

        selected_sum = doc_embeddings[selected[0]] / n

        while len(selected) < n:
            # Compute mean of dot products of all selected documents and all other documents
            similarities = selected_sum @ doc_embeddings.T
            # Mask documents that are already selected
            similarities[selected] = torch.inf
            # Select the document with the lowest total similarity score
            index_unselected = int(torch.argmin(similarities).item())
            selected.append(index_unselected)
            # It's enough just to add to the selected vectors because dot product is distributive
            # It's divided by n for numerical stability
            selected_sum += doc_embeddings[index_unselected] / n

        ranked_docs: List[RetrieverResult] = [documents[i] for i in selected]

        return ranked_docs

    def _embed_and_normalize(
        self, query: str, texts_to_embed: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute embeddings for query and documents, with optional normalization.

        Args:
            query (str): The search query.
            texts_to_embed (List[str]): List of texts to embed.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Document embeddings (shape: [num_docs, embedding_dim])
                - Query embedding (shape: [1, embedding_dim])

        Note:
            If similarity metric is "cosine", embeddings are normalized to unit length.
        """
        # Calculate embeddings
        doc_embeddings = self.model.encode(texts_to_embed, convert_to_tensor=True)  # type: ignore[attr-defined]
        query_embedding = self.model.encode(
            [self.query_prefix + query + self.query_suffix], convert_to_tensor=True
        )  # type: ignore[attr-defined]

        # Normalize embeddings to unit length for computing cosine similarity
        if self.similarity == DiversityRankingSimilarity.COSINE:
            doc_embeddings /= torch.norm(doc_embeddings, p=2, dim=-1).unsqueeze(-1)
            query_embedding /= torch.norm(query_embedding, p=2, dim=-1).unsqueeze(-1)
        return doc_embeddings, query_embedding

    def _maximum_margin_relevance(
        self,
        query: str,
        documents: List[RetrieverResult],
        lambda_threshold: float,
        top_k: int,
    ) -> List[RetrieverResult]:
        """
        Rerank documents using the Maximum Margin Relevance (MMR) strategy.

        MMR balances relevance and diversity by computing a score for each document that combines:
        - Relevance to the query
        - Diversity from already selected documents

        The score is computed as:
            MMR_score = lambda * relevance_score - (1 - lambda) * max_similarity_to_selected

        Args:
            query (str): The search query.
            documents (List[RetrieverResult]): List of documents to rerank.
            lambda_threshold (float): Trade-off parameter between relevance and diversity (0.0 to 1.0).
            top_k (int): Maximum number of documents to return.

        Returns:
            List[RetrieverResult]: Reranked documents ordered by MMR scores.

        Note:
            Based on the paper: "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries"
            by Carbonell and Goldstein (1998).
        """

        texts_to_embed = self._prepare_texts_to_embed(documents)
        doc_embeddings, query_embedding = self._embed_and_normalize(
            query, texts_to_embed
        )
        top_k = top_k if top_k else len(documents)

        selected: List[int] = []
        query_similarities_as_tensor = query_embedding @ doc_embeddings.T
        query_similarities = query_similarities_as_tensor.reshape(-1)
        idx = int(torch.argmax(query_similarities))
        selected.append(idx)
        while len(selected) < top_k:
            best_idx = None
            best_score = -float("inf")
            for idx, _ in enumerate(documents):
                if idx in selected:
                    continue
                relevance_score = query_similarities[idx]
                diversity_score = max(
                    doc_embeddings[idx] @ doc_embeddings[j].T for j in selected
                )
                mmr_score = (
                    lambda_threshold * relevance_score
                    - (1 - lambda_threshold) * diversity_score
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            if best_idx is None:
                raise ValueError(
                    "No best document found, check if the documents list contains any documents."
                )
            selected.append(best_idx)

        return [documents[i] for i in selected]

    @staticmethod
    def _check_lambda_threshold(
        lambda_threshold: float, strategy: DiversityRankingStrategy
    ) -> None:
        """
        Validate the lambda threshold parameter for MMR strategy.

        Args:
            lambda_threshold (float): The lambda threshold to validate.
            strategy (DiversityRankingStrategy): The current ranking strategy.

        Raises:
            ValueError: If strategy is MMR and lambda_threshold is not between 0 and 1.
        """
        if (
            strategy == DiversityRankingStrategy.MAXIMUM_MARGIN_RELEVANCE
        ) and not 0 <= lambda_threshold <= 1:
            raise ValueError(
                f"lambda_threshold must be between 0 and 1, but got {lambda_threshold}."
            )

    def _rerank(
        self,
        query: RetrieverQueryType,
        results: list[list[RetrieverResult]] | list[RetrieverResult],
        top_k: int = 10,
        lambda_threshold: Optional[float] = None,
        **kwargs,
    ) -> list[RerankerResult]:
        """
        Rerank the results using either Maximum Margin Relevance (MMR) or Greedy Diversity Order strategy.

        The method uses sentence transformers to compute embeddings and then applies one of two strategies:
        1. Maximum Margin Relevance (MMR): Balances relevance and diversity using a lambda threshold
        2. Greedy Diversity Order: Maximizes diversity while maintaining relevance

        Args:
            query (RetrieverQueryType): The search query.
            results (list[list[RetrieverResult]] | list[RetrieverResult]): The results to rerank.
            top_k (int): The maximum number of results to return.
            lambda_threshold (Optional[float]): The trade-off parameter between relevance and diversity.
                                              Only used when strategy is "maximum_margin_relevance".
                                              A value closer to 1 favors relevance, while a value closer to 0 favors diversity.
            **kwargs: Additional arguments for the reranker.

        Returns:
            list[RerankerResult]: The reranked results, where each result contains:
                - id: The document identifier
                - score: The relevance score
                - document: The document content
                - metadata: Additional document metadata

        Raises:
            ValueError: If the top_k value is less than or equal to 0 or greater than the number of results.
            RuntimeError: If the component has not been warmed up.
        """
        if self.model is None:
            error_msg = (
                "The component SentenceTransformersDiversityRanker wasn't warmed up. "
                "Run 'warm_up()' before calling 'run()'."
            )
            raise RuntimeError(error_msg)

        if not results:
            return []

        if top_k is None:
            top_k = self.top_k
        elif not 0 < top_k <= len(results):
            raise ValueError(
                f"top_k must be between 1 and {len(results)}, but got {top_k}"
            )

        if self.strategy == DiversityRankingStrategy.MAXIMUM_MARGIN_RELEVANCE:
            if lambda_threshold is None:
                lambda_threshold = self.lambda_threshold
            self._check_lambda_threshold(lambda_threshold, self.strategy)
            re_ranked_docs = self._maximum_margin_relevance(
                query=query,
                documents=results,
                lambda_threshold=lambda_threshold,
                top_k=top_k,
            )
        else:
            re_ranked_docs = self._greedy_diversity_order(
                query=query, documents=results
            )

        return [
            RerankerResult(
                id=doc.id,
                score=doc.score,
                document=doc.document,
                metadata=doc.metadata,
            )
            for doc in re_ranked_docs
        ]
