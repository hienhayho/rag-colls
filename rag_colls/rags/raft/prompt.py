from enum import Enum
from pydantic import BaseModel

CONTEXTUAL_BOOST_SYSTEM_PROMPT = """{{
    "Persona": "You are a helpful assistant that summarizes the content of the provided paragraph in a single sentence.",
    "Instructions": [
        "Be sure to include all the main ideas from the provided paragraph.",
        "In your summary sentence, mention which document the paragraph is from.",
        "Do not include unnecessary phrases such as: 'This is a summary of the paragraph', 'The paragraph talks about', etc."
    ],
    "Note": "You will be rewarded with 10,000 points if you do a good job."
}}"""

GENERATE_QUESTION_SYSTEM_PROMPT = """{{
    "Persona": "You are an expert question generator. Your task is to create questions based on the provided document content.",
    "Instructions": [
        "You will be given a passage that includes the document name and a portion of its content.",
        "Generate up to 5 questions that can be answered solely from the provided content.",
        "Each question must explicitly mention the name of the document.",
        "Your questions must be in document's language.",
    ],
    "Output format" : "Return only the list of questions in the following JSON format:
```json
{
    "questions": ["Question 1", "Question 2", "..."]
}
```",
    "Note": "You will be rewarded with 10,000 points if you do a good job."
}}"""

GENERATE_REASONING_ANSWER_SYSTEM_PROMPT = """{{
    "Persona": "You are a reasoning assistant. Your task is to answer the given question using the provided context.",
    "Instructions": [
        "First, provide a step-by-step reasoning about how to answer the question based on the context.",
        "Make sure that only the quoted parts are copied verbatim from the context; the rest of the reasoning must be in your own words.",
    ],
    "Output format" : "Return exactly in the format: <REASON>: "your-reason"\\n<ANSWER>: "your-answer" without any additional text.",
    "Note": [
        "Be thorough and logical in the reasoning.",
        "Do not include any other formatting or explanation outside the instructions.",
        "You will be rewarded with 10,000 points if you do a good job."
    ]
}}"""


class PromptMode(BaseModel):
    json_version: str = ""
    plain_text_version: str = ""
    xml_version: str = ""
    yaml_version: str = ""
    markdown_version: str = ""


class PromptModeEnum(str, Enum):
    JSON = "json"
    PLAIN_TEXT = "plain_text"
    XML = "xml"
    YAML = "yaml"
    MARKDOWN = "markdown"

    def __str__(self):
        return self.value


class Prompt(BaseModel):
    system_prompt: PromptMode
    user_prompt: PromptMode


def get_prompt(
    prompt: Prompt, mode: PromptModeEnum = PromptModeEnum.JSON
) -> tuple[str, str]:
    """
    Get the prompt in the specified mode.
    """
    system_prompt, user_prompt = None, None
    if mode == PromptModeEnum.JSON:
        system_prompt, user_prompt = (
            prompt.system_prompt.json_version,
            prompt.user_prompt.json_version,
        )
    elif mode == PromptModeEnum.PLAIN_TEXT:
        system_prompt, user_prompt = (
            prompt.system_prompt.plain_text_version,
            prompt.user_prompt.plain_text_version,
        )
    elif mode == PromptModeEnum.XML:
        system_prompt, user_prompt = (
            prompt.system_prompt.xml_version,
            prompt.user_prompt.xml_version,
        )
    elif mode == PromptModeEnum.YAML:
        system_prompt, user_prompt = (
            prompt.system_prompt.yaml_version,
            prompt.user_prompt.yaml_version,
        )
    elif mode == PromptModeEnum.MARKDOWN:
        system_prompt, user_prompt = (
            prompt.system_prompt.markdown_version,
            prompt.user_prompt.markdown_version,
        )
    else:
        raise ValueError("Invalid mode")

    return system_prompt, user_prompt


gen_data_prompt = Prompt(
    system_prompt=PromptMode(
        plain_text_version="""You are a reasoning assistant. Your task is to provide logical reasoning to answer the given question based on provided context.

Instruction:  Given the question, context and answer above, provide a logical reasoning for that answer. Please use the format of: <REASON>: {{reason}} <ANSWER>: {{answer}}.""",
        markdown_version="""## Persona
- You are a reasoning assistant. Your task is to provide logical reasoning to answer the given question based on provided context.

## Instruction:
- Given the question, context and answer above, provide a logical reasoning for that answer.
- Please use the format of: <REASON>: {{reason}} <ANSWER>: {{answer}}.
""",
        yaml_version="""Persona
- You are a reasoning assistant. Your task is to provide logical reasoning to answer the given question based on provided context.

Instruction:
- Given the question, context and answer above, provide a logical reasoning for that answer.
- Please use the format of: <REASON>: {{reason}} <ANSWER>: {{answer}}.""",
        xml_version="""<Persona>You are a reasoning assistant. Your task is to provide logical reasoning to answer the given question based on provided context.</Persona>

<Instruction>Given the question, context and answer above, provide a logical reasoning for that answer. Please use the format of: <REASON>: {{reason}} <ANSWER>: {{answer}}.</Instruction>""",
        json_version="""{{
  "Persona": "You are a reasoning assistant. Your task is to provide logical reasoning to answer the given question based on provided context.",
  "Instructions": [
    "Analyze the question, context, and the given answer carefully.",
    "Provide a logical reasoning explaining why the answer is appropriate based on the context.",
    "Use the following output format exactly:\\n\\n<REASON>: {{reason}}\\n<ANSWER>: {{answer}}"
  ],
  "Output format": "Return exactly in the format: <REASON>: {{reason}}\\n<ANSWER>: {{answer}} without any additional text.
}}""",
    ),
    user_prompt=PromptMode(
        plain_text_version="""Question: {query}
===================
Context: {context}""",
        json_version="""{{
  "Question": "{query}",
  "Context": "{context}"
}}""",
        markdown_version="""## Question
{query}

## Context
{context}""",
        yaml_version="""Question
{query}

Context
{context}""",
        xml_version="""<Question>{query}</Question>
<Context>{context}</Context>""",
    ),
)
