from pydantic import BaseModel
from rag_colls.types.llm import Message
from rag_colls.llms.vllm_llm import VLLM


# Define the response format
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


llm = VLLM(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    gpu_memory_utilization=0.5,
    dtype="half",
    download_dir="./model_cache",
)


messages = [
    Message(role="system", content="Extract the event information."),
    Message(
        role="user", content="Alice and Bob are going to a science fair on Friday."
    ),
]

response = llm.complete(
    messages=messages,
    response_format=CalendarEvent,
    max_tokens=512,
)

# Print the response
result = CalendarEvent.model_validate_json(response.content)
print(result)
print("==========================")
print(response.usage)
