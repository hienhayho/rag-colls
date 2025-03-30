from pydantic import BaseModel, Field


class Embedding(BaseModel):
    """Base class for all embeddings."""

    embedding: list[float] = Field(
        ..., description="List of floats representing the embedding"
    )
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the embedding"
    )

    def _get_metadata(self):
        metadatas = []
        for key, value in self.metadata.items():
            metadatas.append(f"{key}={value}")
        return ", ".join(metadatas)

    def __str__(self):
        metadata_str = self._get_metadata()
        return f"Embedding(dimensions={len(self.embedding)}, {metadata_str})"

    def __repr__(self):
        return self.__str__()
