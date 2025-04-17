import bm25s
import shutil
from pathlib import Path
from loguru import logger

from rag_colls.core.base.database.bm25 import BaseBM25RetrieverProvider
from rag_colls.types.retriever import (
    RetrieverResult,
    RetrieverQueryType,
    RetrieverIngestInput,
)


class BM25s(BaseBM25RetrieverProvider):
    """
    Wrapper for the BM25s library: `https://github.com/xhluca/bm25s`
    """

    def __init__(self, save_dir: str | Path):
        self.save_dir = save_dir
        self.reloaded_retriever = None

        if Path(self.save_dir).exists():
            self.reloaded_retriever = bm25s.BM25.load(
                save_dir=save_dir, load_corpus=True
            )

    def _test_connection(self):
        """
        Test the connection to the BM25s database.
        """
        # NOTE: Since BM25s is a local database, we don't need to test the connection.
        return True

    def _build_corpus(self, documents: list[RetrieverIngestInput]):
        corpus_json = [
            {"text": doc.document, "metadata": {"id": doc.id, **doc.metadata}}
            for doc in documents
        ]
        return corpus_json

    def _clean_resource(self):
        """
        Clean the BM25s resource.
        """
        if Path(self.save_dir).exists():
            shutil.rmtree(self.save_dir, ignore_errors=True)
            logger.debug(f"Cleaned up BM25s database at {self.save_dir}")
        else:
            logger.debug(f"BM25s database at {self.save_dir} does not exist.")

    def _index_documents(self, documents: list[RetrieverIngestInput], **kwargs):
        """
        Ingest documents into the BM25s database.

        Args:
            documents (list[Document]): List of documents to ingest.
        """
        corpus_json = self._build_corpus(documents)
        retriever = bm25s.BM25(corpus=corpus_json)

        corpus_text = [doc["text"] for doc in corpus_json]

        corpus_tokens = bm25s.tokenize(corpus_text)

        retriever.index(corpus_tokens)
        retriever.save(self.save_dir)

    def _add_documents_to_index(self, documents: list[RetrieverIngestInput], **kwargs):
        """
        Add documents to the existing BM25s index by re-indexing the entire corpus.

        Refer to: `https://github.com/xhluca/bm25s/discussions/20`

        Args:
            documents (list[Document]): List of documents to ingest.
        """
        logger.debug("Re-indexing ...")

        reloaded_retriever = bm25s.BM25.load(self.save_dir, load_corpus=True)
        old_corpus = reloaded_retriever.corpus
        new_corpus = self._build_corpus(documents)

        merged_corpus = old_corpus + new_corpus

        # NOTE: If the re-indexing fails or an exception occurs during indexing,
        # the previous data will be lost. We should have a backup of the old data.
        shutil.rmtree(self.save_dir, ignore_errors=True)

        self._index_documents(merged_corpus, **kwargs)

    def _add_documents(self, documents: list[RetrieverIngestInput], **kwargs):
        """
        Ingest documents into the BM25s database.

        Args:
            documents (list[Document]): List of documents to ingest.
        """
        if Path(self.save_dir).exists():
            self._add_documents_to_index(documents, **kwargs)
        else:
            self._index_documents(documents, **kwargs)

    def _search(
        self, query: RetrieverQueryType, top_k: int = 5, **kwargs
    ) -> list[RetrieverResult]:
        """
        Perform a search using the BM25s retriever.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to return.

        Returns:
            list[RetrieverResult]: List of retrieved documents with their metadata.
        """
        assert isinstance(query, str), "Query must be a string."

        query_tokens = bm25s.tokenize(query)
        if self.reloaded_retriever is None:
            self.reloaded_retriever = bm25s.BM25.load(self.save_dir, load_corpus=True)

        results, scores = self.reloaded_retriever.retrieve(query_tokens, k=top_k)

        search_results: list[RetrieverResult] = []

        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            search_results.append(
                RetrieverResult(
                    id=doc["metadata"]["id"],
                    score=score,
                    document=doc["text"],
                    metadata=doc["metadata"],
                )
            )

        # Normalize scores to [0, 1]
        max_score = max(scores[0])
        min_score = min(scores[0])

        for result in search_results:
            result.score = (
                (result.score - min_score) / (max_score - min_score)
                if max_score != min_score
                else 0.0
            )

        return search_results
