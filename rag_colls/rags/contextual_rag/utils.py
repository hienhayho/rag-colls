from rag_colls.types.llm import Message
from rag_colls.types.core.document import Document
from rag_colls.core.base.llms.base import BaseCompletionLLM

from .prompt import CONTEXTUAL_PROMPT


def gen_contextual_chunk(
    chunk: Document,
    whole_document: Document,
    llm: BaseCompletionLLM,
    contextual_prompt_template: str = CONTEXTUAL_PROMPT,
) -> Document:
    """
    Generate a contextual document using the provided LLM.

    Args:
        chunk (Document): The chunk of text to be contextualized.
        whole_document (Document): The whole document containing the context.
        llm (BaseCompletionLLM): The language model used for generation.
        contextual_prompt_template (str): The template for the contextual prompt.

    Returns:
        Document: The generated contextual document.
    """
    messages = [
        Message(
            role="user",
            content=contextual_prompt_template.format(
                CHUNK_CONTENT=chunk.document, WHOLE_DOCUMENT=whole_document.document
            ),
        )
    ]

    response = llm.complete(messages)

    return Document(
        id=chunk.id,
        document=response.content + "\n\n" + chunk.document,
        metadata=chunk.metadata,
    )
