from rag_colls.types.llm import Message
from rag_colls.llms.vllm_llm import VLLM

llm = VLLM(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    gpu_memory_utilization=0.5,
    dtype="half",
    download_dir="./model_cache",
)

messages = [
    Message(role="user", content="What is the capital of France?"),
]

response = llm.complete(messages=messages, max_tokens=512)

print(response.content)
print("==========================")
print(response.usage)
