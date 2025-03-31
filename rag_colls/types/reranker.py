from rag_colls.types.retriever import RetrieverResult


class RerankerResult(RetrieverResult):
    def __str__(self):
        return f"RerankerResult(score={self.score}, metadata={self.metadata})"
