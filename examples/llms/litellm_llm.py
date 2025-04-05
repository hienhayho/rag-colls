from rag_colls.types.llm import Message
from rag_colls.llms.litellm_llm import LiteLLM

llm = LiteLLM(model_name="openai/gpt-4o-mini")

messages = [
    Message(role="user", content="What is the capital of France?"),
]

response = llm.complete(messages=messages)

print(response.content)
print("==========================")
print(response.usage)
