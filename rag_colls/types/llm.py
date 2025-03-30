from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(
        ..., description="Role of the message sender (user, assistant, system)."
    )
    content: str = Field(..., description="Content of the message.")

    def __str__(self):
        return f"Message(role={self.role}, content={self.content})"

    def __repr__(self):
        return self.__str__()


class LLMUsage(BaseModel):
    """
    Class to represent the usage of LLM resources.
    """

    prompt_tokens: int = Field(0, description="Number of tokens in the prompt.")
    completion_tokens: int = Field(0, description="Number of tokens in the completion.")
    total_tokens: int = Field(0, description="Total number of tokens used.")

    def __str__(self):
        return f"LLMUsage(prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens}, total_tokens={self.total_tokens})"

    def __repr__(self):
        return self.__str__()


class LLMOutput(BaseModel):
    """
    Base class for LLM outputs.
    """

    content: str = Field(..., description="Text of the LLM output.")
    usage: LLMUsage = Field(
        LLMUsage(), description="Usage information for the LLM output."
    )

    def __str__(self):
        return f"LLMOutput(content={self.content}, usage={self.usage})"

    def __repr__(self):
        return self.__str__()
