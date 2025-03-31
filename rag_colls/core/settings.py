from pydantic import BaseModel, Field, ConfigDict

from rag_colls.loggers.loguru import LoguruLogger
from rag_colls.core.base.loggers.base import BaseLogger
from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.core.base.embeddings.base import BaseEmbedding

from rag_colls.llms.litellm_llm import LiteLLM
from rag_colls.embeddings.openai_embedding import OpenAIEmbedding


class RagCollsSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    logger: BaseLogger = Field(..., description="Logger to use in the application")
    embed_model: BaseEmbedding = Field(
        ..., description="Embedding model to use in the application"
    )
    completion_llm: BaseCompletionLLM = Field(
        ..., description="Completion LLM to use in the application"
    )

    def __str__(self):
        return f"RagCollsSettings(logger={self.logger}, embed_model={self.embed_model}, completion_llm={self.completion_llm})"


GlobalSettings = RagCollsSettings(
    logger=LoguruLogger(), embed_model=OpenAIEmbedding(), completion_llm=LiteLLM()
)

GlobalSettings.logger.info("GlobalSettings initialized ...")
print(GlobalSettings)
print("===========")
