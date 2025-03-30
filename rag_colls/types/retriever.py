from pydantic import BaseModel, Field


class RetrieverIngestInput(BaseModel):
    id: str = Field(
        ..., description="The unique identifier for the document to be ingested."
    )
    document: str = Field(
        ..., description="The content of the document to be ingested."
    )
    embedding: list[float] = Field(
        ..., description="The embedding vector of the document to be ingested."
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata associated with the document to be ingested.",
    )

    def __str__(self):
        return f"IngestInput(id={self.id}, dimensions={len(self.embedding)}, metadata={self.metadata})"

    def __repr__(self):
        return self.__str__()


class RetrieverResult(BaseModel):
    """
    A class representing the result of a retrieval operation.
    """

    id: str = Field(..., description="The unique identifier of the retrieved document.")
    score: float = Field(
        ..., description="The score or relevance of the retrieved document."
    )
    document: str = Field(..., description="The content of the retrieved document.")
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata associated with the retrieved document.",
    )

    def __str__(self):
        return f"RetrieveResult(id={self.id}, score={self.score}, document={self.document}, metadata={self.metadata})"

    def __repr__(self):
        return self.__str__()
