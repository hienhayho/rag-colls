from typing import Type
from pydantic import BaseModel
from abc import ABC, abstractmethod
from rag_colls.types.llm import Message
from rag_colls.types.llm import LLMOutput


class BaseCompletionLLM(ABC):
    @abstractmethod
    def _complete(
        self,
        messages: list[Message],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMOutput:
        """
        Generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            response_format (Type[BaseModel] | None): The JSON format of the response.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            LLMOutput: The output of the model containing the generated content and usage information.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the model.

        Returns:
            str: The name of the model.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _batch_complete(
        self,
        messages: list[list[Message]],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> list[LLMOutput]:
        """
        Generates completions for a batch of messages.

        Args:
            messages (list[list[Message]]): List of batches of messages to be sent to the model.
            response_format (Type[BaseModel] | None): The JSON format of the response.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            list[LLMOutput]: List of outputs from the model containing generated content and usage information.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _is_support_json_output(self) -> bool:
        """
        Checks if the model supports JSON output.

        Returns:
            bool: True if the model supports JSON output, False otherwise.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def _acomplete(
        self,
        messages: list[Message],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMOutput:
        """
        Asynchronously generates a completion based on the provided messages.

         Args:
            messages (list[Message]): List of messages to be sent to the model.
            response_format (Type[BaseModel] | None): The JSON format of the response.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            LLMOutput: The output of the model containing the generated content and usage information.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def complete(
        self,
        messages: list[Message],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMOutput:
        """
        Generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            response_format (Type[BaseModel] | None): The JSON format of the response.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            str: The generated completion.
        """
        if not self._is_support_json_output() and response_format:
            raise ValueError(
                "This model does not support JSON output. Please set response_format to None."
            )

        result = self._complete(messages, response_format=response_format, **kwargs)

        if response_format:
            try:
                response_format.model_validate_json(result.content)
            except Exception as e:
                raise ValueError(f"Invalid response format: {e}") from e

        return result

    def batch_complete(
        self,
        messages: list[list[Message]],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> list[LLMOutput]:
        """
        Generates completions for a batch of messages.

        Args:
            messages (list[list[Message]]): List of batches of messages to be sent to the model.
            response_format (Type[BaseModel] | None): The JSON format of the response.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            list[LLMOutput]: List of outputs from the model containing generated content and usage information.
        """
        if not self._is_support_json_output() and response_format:
            raise ValueError(
                "This model does not support JSON output. Please set response_format to None."
            )

        result = self._batch_complete(
            messages, response_format=response_format, **kwargs
        )

        if response_format:
            try:
                for r in result:
                    response_format.model_validate_json(r.content)
            except Exception as e:
                raise ValueError(f"Invalid response format: {e}") from e

        return result

    async def abatch_complete(
        self,
        messages: list[list[Message]],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> list[LLMOutput]:
        """
        Asynchronously generates completions for a batch of messages.

        Args:
            messages (list[list[Message]]): List of batches of messages to be sent to the model.
            response_format (Type[BaseModel] | None): The JSON format of the response.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            list[LLMOutput]: List of outputs from the model containing generated content and usage information.
        """
        if not self._is_support_json_output() and response_format:
            raise ValueError(
                "This model does not support JSON output. Please set response_format to None."
            )

        result = await self._batch_complete(
            messages, response_format=response_format, **kwargs
        )

        if response_format:
            try:
                for r in result:
                    response_format.model_validate_json(r.content)
            except Exception as e:
                raise ValueError(f"Invalid response format: {e}") from e

        return result

    async def acomplete(
        self,
        messages: list[Message],
        response_format: Type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMOutput:
        """
        Asynchronously generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            response_format (Type[BaseModel] | None): The JSON format of the response.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            str: The generated completion.
        """
        if not self._is_support_json_output() and response_format:
            raise ValueError(
                "This model does not support JSON output. Please set response_format to None."
            )

        result = await self._acomplete(messages, response_format, **kwargs)
        if response_format:
            try:
                response_format.model_validate_json(result.content)
            except Exception as e:
                raise ValueError(f"Invalid response format: {e}") from e

        return result
