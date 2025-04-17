from pydantic import BaseModel, Field, ConfigDict

from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.core.base.embeddings.base import BaseEmbedding

from rag_colls.llms.litellm_llm import LiteLLM
from rag_colls.embeddings.openai_embedding import OpenAIEmbedding


class RagCollsSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embed_model: BaseEmbedding = Field(
        ..., description="Embedding model to use in the application"
    )
    completion_llm: BaseCompletionLLM = Field(
        ..., description="Completion LLM to use in the application"
    )

    def __str__(self):
        return f"RagCollsSettings(embed_model={self.embed_model}, completion_llm={self.completion_llm})"


GlobalSettings = RagCollsSettings(
    embed_model=OpenAIEmbedding(), completion_llm=LiteLLM()
)
