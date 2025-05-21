from rag_colls.types.llm import Message
from rag_colls.types.core.document import Document
from rag_colls.core.base.llms.base import BaseCompletionLLM

from .prompt import CONTEXTUAL_BOOST_SYSTEM_PROMPT


def get_contextual_boost_document(
    document: Document,
    llm: BaseCompletionLLM,
    system_prompt: str = CONTEXTUAL_BOOST_SYSTEM_PROMPT,
    **kwargs,
) -> Document:
    """
    Generate a contextual boost for the given text.

    Args:
        text (str): The input text to generate a contextual boost for.
        llm (BaseCompletionLLM): The LLM to use for generating the contextual boost.
        system_prompt (str): The system prompt to use.

    Returns:
        Document: The generated contextual boost as a Document object.
    """
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=document.document),
    ]

    response = llm.complete(
        messages=messages,
        **kwargs,
    )

    return Document(
        id=document.id,
        document=f"{response.content}\n\n{document.document}",
        metadata=document.metadata,
    )
