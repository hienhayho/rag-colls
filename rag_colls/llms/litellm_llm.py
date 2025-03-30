from litellm import completion, acompletion

from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.types.llm import Message, LLMOutput, LLMUsage


class LiteLLM(BaseCompletionLLM):
    """
    A lightweight wrapper for the litellm library.

    litellm provide many models from openai, anthropic, google, etc..
    """

    default_model = "openai/gpt-4o-mini"

    def __init__(self, model_name: str | None = None):
        """
        Initialize the LiteLLM class.

        Args:
            model_name (str): The name of the model to use.
        """
        self.model_name = model_name or self.default_model

    def _complete(self, messages: list[Message], **kwargs) -> LLMOutput:
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

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
