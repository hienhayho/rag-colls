from json_repair import repair_json
from pydantic import BaseModel, Field

from rag_colls.types.llm import Message
from rag_colls.core.base.llms.base import BaseCompletionLLM


class LLMAsAJudgeResponse(BaseModel):
    approved: bool = Field(
        ...,
        description="Whether the answer is approved or not base on the query and context.",
    )


def llm_as_a_judge_inference(
    llm: BaseCompletionLLM,
    *,
    queries: list[str],
    contexts: list[str],
    referenced_answers: list[str],
    answers: list[str],
) -> list[LLMAsAJudgeResponse]:
    messages = [
        [
            Message(
                role="system",
                content="""You are an impartial and meticulous judge tasked with evaluating whether a given answer appropriately addresses a query, using the provided context as the sole source of truth.

Your responsibilities:

- Carefully read the **query**, **context**, **referenced answer**, and **answer**.

- Use only the context to evaluate the answer — do not rely on external knowledge or assumptions.

- Take the **referenced answer** into account as a baseline or comparison when assessing the quality and accuracy of the answer.

- Determine if the answer is:
  • Factually correct
  • Fully supported by the context
  • Relevant and responsive to the query

Avoid hallucination. Be fair, objective, and concise in your evaluation.""",
            ),
            Message(
                role="user",
                content=f"""Query: {query}

================
Context: {context}

================
Referenced Answer: {referenced_answer}

================
Answer: {answer}
""",
            ),
        ]
        for query, context, referenced_answer, answer in zip(
            queries, contexts, referenced_answers, answers
        )
    ]

    responses = llm.batch_complete(
        messages=messages, response_format=LLMAsAJudgeResponse
    )

    return [
        LLMAsAJudgeResponse.model_validate_json(repair_json(r.content))
        for r in responses
    ]
