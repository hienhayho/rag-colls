from pydantic import BaseModel, Field

from rag_colls.types.llm import LLMUsage
from rag_colls.types.retriever import RetrieverResult


class SearchOutput(BaseModel):
    content: str = Field(
        ...,
        description="The final response of the search result.",
    )
    usage: LLMUsage = Field(
        ...,
        description="The usage information of the LLM.",
    )
    retrieved_results: list[RetrieverResult] = Field(
        [],
        description="The retrieved results from the search.",
    )
    retrieved_time: float = Field(
        0,
        description="The time taken to retrieve the results.",
    )
    generation_time: float = Field(
        0,
        description="The time taken to generate the response.",
    )
