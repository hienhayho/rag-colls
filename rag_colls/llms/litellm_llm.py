from typing import Type
from loguru import logger
from pydantic import BaseModel
from litellm import (
    completion,
    batch_completion,
    acompletion,
    get_supported_openai_params,
    supports_response_schema,
)

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

    def _is_support_json_output(self):
        """
        Check if the model supports JSON output.

        Returns:
            bool: True if the model supports JSON output, False otherwise.
        """
        try:
            assert "response_format" in get_supported_openai_params(self.model_name)
            assert supports_response_schema(self.model_name)
            return True

        except AssertionError:
            logger.warning(
                f"Model {self.model_name} does not support JSON output. "
                "Supported models here: https://docs.litellm.ai/docs/completion/json_mode#pass-in-json_schema"
            )
            return False

    def _complete(
        self,
        messages: list[Message],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMOutput:
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # only get params from kwargs which are in completion.__annotations__
        kwargs = {k: v for k, v in kwargs.items() if k in completion.__annotations__}
        response = completion(
            model=self.model_name,
            messages=formatted_messages,
            response_format=response_format,
            **kwargs,
        )
        return LLMOutput(
            content=response.choices[0].message.content,
            usage=LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    def _batch_complete(
        self,
        messages: list[list[Message]],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> list[LLMOutput]:
        formatted_messages = [
            [{"role": msg.role, "content": msg.content} for msg in message]
            for message in messages
        ]

        # only get params from kwargs which are in completion.__annotations__
        kwargs = {
            k: v for k, v in kwargs.items() if k in batch_completion.__annotations__
        }
        responses = batch_completion(
            model=self.model_name,
            messages=formatted_messages,
            response_format=response_format,
            **kwargs,
        )

        return [
            LLMOutput(
                content=response.choices[0].message.content,
                usage=LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
            )
            for response in responses
        ]

    async def _acomplete(
        self,
        messages: list[Message],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMOutput:
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        response = await acompletion(
            model=self.model_name,
            messages=formatted_messages,
            response_format=response_format,
            **kwargs,
        )

        return LLMOutput(
            content=response.choices[0].message.content,
            usage=LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )
