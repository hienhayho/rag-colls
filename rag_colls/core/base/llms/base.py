from abc import ABC, abstractmethod
from rag_colls.types.llm import Message
from rag_colls.types.llm import LLMOutput


class BaseCompletionLLM(ABC):
    @abstractmethod
    def _complete(self, messages: list[Message], **kwargs) -> LLMOutput:
        """
        Generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            LLMOutput: The output of the model containing the generated content and usage information.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def _acomplete(self, messages: list[Message], **kwargs) -> LLMOutput:
        """
        Asynchronously generates a completion based on the provided messages.

         Args:
            messages (list[Message]): List of messages to be sent to the model.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            LLMOutput: The output of the model containing the generated content and usage information.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def complete(self, messages: list[Message], **kwargs) -> LLMOutput:
        """
        Generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            str: The generated completion.
        """
        return self._complete(messages, **kwargs)

    async def acomplete(self, messages: list[Message], **kwargs) -> LLMOutput:
        """
        Asynchronously generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            **kwargs: Additional keyword arguments for the completion function.

        Returns:
            str: The generated completion.
        """
        return await self._acomplete(messages, **kwargs)
