import argparse
import polars as pl
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, Field

from rag_colls.types.llm import Message
from rag_colls.core.utils import load_chunks
from rag_colls.llms.litellm_llm import LiteLLM
from rag_colls.types.core.document import Document

from rag_colls.rags.raft import (
    get_prompt,
    gen_data_prompt,
    PromptModeEnum,
    GENERATE_QUESTION_SYSTEM_PROMPT,
    GENERATE_REASONING_ANSWER_SYSTEM_PROMPT,
)

from ingest import get_rag
import litellm


class GenerateQuestionOutput(BaseModel):
    questions: list[str] = Field(..., description="List of generated questions")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning data.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--f", nargs="+", type=str, help="File paths to load chunks from"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top-k documents to retrieve for each question",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for the fine-tuning data",
        default="data.jsonl",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=PromptModeEnum.__members__.values(),
        default=PromptModeEnum.JSON,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model name for generating the fine-tuning data",
    )
    return parser.parse_args()


def generate_questions(chunks: list[Document], llm: LiteLLM):
    messages = [
        [
            Message(role="system", content=GENERATE_QUESTION_SYSTEM_PROMPT),
            Message(role="user", content=chunk.document),
        ]
        for chunk in chunks
    ]

    response = llm.batch_complete(
        messages=messages,
        response_format=GenerateQuestionOutput,
    )

    return [
        GenerateQuestionOutput.model_validate_json(r.content).questions
        for r in response
    ]


def generate_answer(
    questions: list[str],
    context: Document,
    llm: LiteLLM,
):
    messages = [
        [
            Message(role="system", content=GENERATE_REASONING_ANSWER_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Context: {context.document}\n==================== \n Question: {question}",
            ),
        ]
        for question in questions
    ]

    response = llm.batch_complete(messages=messages)
    return [response.content for response in response]


def main(args):
    llm = LiteLLM(model_name=args.model)
    rag = get_rag()

    chunks = load_chunks(file_paths=args.f)
    results = []
    system_prompt, user_prompt = get_prompt(
        prompt=gen_data_prompt, mode=args.prompt_mode
    )

    questions = generate_questions(chunks=chunks, llm=llm)
    for qs, chunk in tqdm(
        zip(questions, chunks), desc="Generating dataset ...", total=len(chunks)
    ):
        answers = generate_answer(
            questions=qs,
            context=chunk,
            llm=llm,
        )
        for q, answer in zip(qs, answers):
            retrieved_contexts = rag.retrieve_db(
                query=q,
                top_k=args.top_k,
            )
            context = "\n".join(
                f"<DOCUMENT>{c.document}</DOCUMENT>" for c in retrieved_contexts
            )
            results.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt.format(
                                query=q,
                                context=context,
                            ),
                        },
                        {"role": "assistant", "content": answer},
                    ]
                }
            )

    df = pl.DataFrame(results)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_ndjson(args.output_path)

    logger.success(f"Fine-tuning data saved to: {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        litellm._turn_on_debug()

    main(args)
