from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

from rag_colls.types.llm import Message
from rag_colls.core.base.llms.base import BaseCompletionLLM


class LlamaIndexLLM(CustomLLM):
    llm: BaseCompletionLLM

    @property
    def metadata(self) -> LLMMetadata:
        """
        Get the metadata for the LlamaIndexLLM.

        Returns:
            LLMMetadata: The metadata for the LlamaIndexLLM.
        """
        return LLMMetadata()

    def __str__(self):
        return f"LlamaIndexLLM(llm={self.llm})"

    def __repr__(self):
        return self.__str__()

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Generate a completion for the given prompt using the LLM.

        Args:
            prompt (str): The input prompt for the LLM.
            **kwargs: Additional keyword arguments.

        Returns:
            CompletionResponse: The response from the LLM.
        """
        messages = [Message(role="user", content=prompt)]
        response = self.llm.complete(messages=messages, **kwargs)

        return CompletionResponse(
            text=response.content,
            additional_kwargs={
                "total_tokens": response.usage.total_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
            },
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """
        Stream completion for the given prompt using the LLM.

        Args:
            prompt (str): The input prompt for the LLM.
            **kwargs: Additional keyword arguments.

        Returns:
            CompletionResponseGen: A generator yielding responses from the LLM.
        """
        messages = [Message(role="user", content=prompt)]
        response = self.llm.complete(messages=messages, **kwargs)

        for token in response.content:
            response += token
            yield CompletionResponse(text=response, delta=token)
