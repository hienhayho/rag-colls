from loguru import logger
from litellm import completion, acompletion

from rag_colls.core.constants import DEFAULT_OPENAI_MODEL
from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.types.llm import Message, LLMOutput, LLMUsage


class LiteLLM(BaseCompletionLLM):
    """
    A lightweight wrapper for the litellm library.

    litellm provide many models from openai, anthropic, google, etc..
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize the LiteLLM class.

        Args:
            model_name (str): The name of the model to use.
        """
        self.model_name = model_name or DEFAULT_OPENAI_MODEL
        logger.info(f"Using LiteLLM with model: {self.model_name}")

    def __str__(self):
        return f"LiteLLM(model_name={self.model_name})"

    def __repr__(self):
        return self.__str__()

    def _complete(self, messages: list[Message], **kwargs) -> LLMOutput:
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # only get params from kwargs which are in completion.__annotations__
        kwargs = {k: v for k, v in kwargs.items() if k in completion.__annotations__}
        response = completion(
            model=self.model_name, messages=formatted_messages, **kwargs
        )
        return LLMOutput(
            content=response.choices[0].message.content,
            usage=LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    async def _acomplete(self, messages: list[Message], **kwargs) -> LLMOutput:
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        response = await acompletion(
            model=self.model_name, messages=formatted_messages, **kwargs
        )

        return LLMOutput(
            content=response.choices[0].message.content,
            usage=LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )
