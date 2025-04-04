import asyncio
from vllm import LLM, SamplingParams
from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.types.llm import Message, LLMOutput, LLMUsage

from transformers import AutoTokenizer


class VLLM(BaseCompletionLLM):
    """
    VLLM wrapper for LLMs.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        dtype: str = "auto",
        quantization: str | None = None,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        download_dir: str = "./model_cache",
        **kwargs,
    ):
        """
        Initialize the VLLM class.

        Args:
            model_name (str): The name of the model to use, can be a huggingface model name or a local path.
            trust_remote_code (bool): Whether to trust remote code. Defaults to `False`.
            tensor_parallel_size (int): Run your model in parallel on multiple GPUs. Defaults to `1`.
            max_model_len (int | None): The maximum model length. Defaults to `None`.
            dtype (str): torch.dtype to use. Defaults to "auto".
            quantization (str | None): The quantization method to use. Defaults to None.
            gpu_memory_utilization (float): The GPU memory utilization. Defaults to 0.9.
            enforce_eager (bool): Whether to enforce eager execution. Defaults to False.
            download_dir (str): The directory to download the model. Defaults to "model_cache".
        """
        kwargs["download_dir"] = download_dir
        self.llm = LLM(
            model=model_name,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _complete(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 1,
        top_p: int = 1,
        top_k: int = -1,
        **kwargs,
    ) -> LLMOutput:
        """
        Generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            max_token (int): Maximum number of tokens to generate. Defaults to `512`.
            temperature (float): Sampling temperature. Defaults to `1`.
            top_p (int): Top-p sampling parameter. Defaults to `1`.
            top_k (int): Top-k sampling parameter. Defaults to `-1`.
            **kwargs: Additional keyword arguments for the completion function. See (https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args) for more details.

        Returns:
            LLMOutput: The output of the model containing the generated content and usage information.
        """
        conversations = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        # only get params from kwargs which in SamplingParams
        kwargs = {
            k: v for k, v in kwargs.items() if k in SamplingParams.__struct_fields__
        }
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            **kwargs,
        )

        response = self.llm.chat(
            conversations, sampling_params=sampling_params, use_tqdm=False
        )

        prompt_tokens = len(response[0].prompt_token_ids)
        completion_tokens = len(response[0].outputs[0].token_ids)

        return LLMOutput(
            content=response[0].outputs[0].text,
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _acomplete(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 1,
        top_p: int = 1,
        top_k: int = -1,
        **kwargs,
    ) -> LLMOutput:
        """
        Asynchronously generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            temperature (float): Sampling temperature. Defaults to `1`.
            top_p (int): Top-p sampling parameter. Defaults to `1`.
            top_k (int): Top-k sampling parameter. Defaults to `-1`.
            **kwargs: Additional keyword arguments for the completion function. See (https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args) for more details.

        Returns:
            LLMOutput: The output of the model containing the generated content and usage information.
        """
        return await asyncio.to_thread(
            self._complete,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

    def complete(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 1,
        top_p: int = 1,
        top_k: int = -1,
        **kwargs,
    ):
        """
        Generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            max_token (int): Maximum number of tokens to generate. Defaults to `512`.
            temperature (float): Sampling temperature. Defaults to `1`.
            top_p (int): Top-p sampling parameter. Defaults to `1`.
            top_k (int): Top-k sampling parameter. Defaults to `-1`.
            **kwargs: Additional keyword arguments for the completion function. See (https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args) for more details.

        Returns:
            LLMOutput: The output of the model containing the generated content and usage information.
        """
        return self._complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

    async def acomplete(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 1,
        top_p: int = 1,
        top_k: int = -1,
        **kwargs,
    ):
        """
        Asynchronously generates a completion based on the provided messages.

        Args:
            messages (list[Message]): List of messages to be sent to the model.
            max_token (int): Maximum number of tokens to generate. Defaults to `512`.
            temperature (float): Sampling temperature. Defaults to `1`.
            top_p (int): Top-p sampling parameter. Defaults to `1`.
            top_k (int): Top-k sampling parameter. Defaults to `-1`.
            **kwargs: Additional keyword arguments for the completion function. See (https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args) for more details.

        Returns:
            LLMOutput: The output of the model containing the generated content and usage information.
        """
        return await self._acomplete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )
