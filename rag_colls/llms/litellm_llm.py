import asyncio
from math import ceil
from tqdm import tqdm
from typing import Type
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import (
    completion,
    acompletion,
    get_supported_openai_params,
    supports_response_schema,
)

from rag_colls.core.constants import DEFAULT_OPENAI_MODEL
from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.types.llm import Message, LLMOutput, LLMUsage

load_dotenv()


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
        batch_size = kwargs.pop("batch_size", 10)
        max_workers = kwargs.pop("max_workers", 4)

        def run_batch(batch: list[list[Message]]) -> list[LLMOutput]:
            return [
                self._complete(message, response_format=response_format, **kwargs)
                for message in batch
            ]

        all_outputs = []
        total_batches = ceil(len(messages) / batch_size)
        batches = [
            messages[i * batch_size : (i + 1) * batch_size]
            for i in range(total_batches)
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_batch, batch) for batch in batches]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Running ..."
            ):
                all_outputs.extend(future.result())

        return all_outputs

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

    async def _abatch_complete(
        self,
        messages: list[list[Message]],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> list[LLMOutput]:
        batch_size = kwargs.pop("batch_size", 10)

        async def run_batch(batch: list[list[Message]]) -> list[LLMOutput]:
            tasks = [
                self._acomplete(
                    message,
                    response_format=response_format,
                    **kwargs,
                )
                for message in batch
            ]
            return await asyncio.gather(*tasks)

        all_outputs = []
        total_batches = ceil(len(messages) / batch_size)

        for i in tqdm_asyncio(range(total_batches), desc="Running async batches"):
            batch = messages[i * batch_size : (i + 1) * batch_size]
            outputs = await run_batch(batch)
            all_outputs.extend(outputs)

        return all_outputs
