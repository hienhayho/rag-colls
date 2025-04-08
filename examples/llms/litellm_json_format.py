from pydantic import BaseModel
from rag_colls.types.llm import Message
from rag_colls.llms.litellm_llm import LiteLLM

# Instance of LiteLLM
llm = LiteLLM(model_name="gpt-4o-mini")

messages = [
    Message(role="system", content="Extract the event information."),
    Message(
        role="user", content="Alice and Bob are going to a science fair on Friday."
    ),
]


# Define the response format
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


response = llm.complete(
    messages=messages,
    response_format=CalendarEvent,
)

# Print the response
result = CalendarEvent.model_validate_json(response.content)
print(result)
print("==========================")
print(response.usage)
