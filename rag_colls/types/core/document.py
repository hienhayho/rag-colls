from uuid import uuid4
from pydantic import BaseModel, Field


def generate_uuid_str() -> str:
    """Generate a new string of UUID."""
    return str(uuid4())


class Document(BaseModel):
    """
    Document class to be used in the RAG pipeline.
    """

    id: str = Field(
        default_factory=generate_uuid_str,
        description="Unique identifier for the document.",
    )
    document: str = Field(..., description="The content of the document.")
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the document."
    )

    def __str__(self):
        return f"Document(id={self.id}, metadata={self.metadata}), document={self.document}"

    def __repr__(self):
        return self.__str__()
