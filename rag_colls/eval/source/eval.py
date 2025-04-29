import os
import json
import litellm
import argparse
from tqdm import tqdm
from loguru import logger
from rich.console import Console
from rich.table import Table

from rag_colls.rags.base import BaseRAG
from rag_colls.core.base.llms.base import BaseCompletionLLM
from rag_colls.eval.source.llm_as_a_judge import llm_as_a_judge_inference

if os.environ.get("DEBUG_LITELLM"):
    litellm._turn_on_debug()


def get_eval_args():
    parser = argparse.ArgumentParser(description="Basic RAG Evaluation")
    parser.add_argument(
        "--f",
        type=str,
        required=True,
        help="Path to the evaluation file",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for evaluation",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization for VLLM",
    )
    parser.add_argument(
        "--o",
        type=str,
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--eval-max-workers",
        type=int,
        default=2,
        help="Number of workers for evaluation",
    )
    return parser.parse_args()


def display_results(
    title: str,
    headers: list[str],
    results: dict,
):
    """
    Display the results in a table format.

    Args:
        title (str): Title of the table.
        headers (list[str]): List of headers for the table.
        results (dict): Dictionary containing the results to display.

    """
    console = Console()
    table = Table(title=title)

    for header in headers:
        table.add_column(header)

    for key, value in results.items():
        table.add_row(key, str(value))

    console.print(table)


def calculate_rr(
    true_idx: str,
    retrieved_idx: list[str],
):
    """
    Calculate the Reciprocal Rank (RR) score.

    Args:
        true_idx (str): The true index of the answer.
        retrieved_idx (list[str]): List of retrieved indices.

    Returns:
        float: RR score.
    """
    try:
        rank = retrieved_idx.index(true_idx) + 1
        return 1 / rank
    except ValueError:
        return 0.0


def eval_search_and_generation(
    rag: BaseRAG,
    eval_file_path: str,
    output_file: str = None,
    eval_llm: BaseCompletionLLM | None = None,
    eval_batch_size: int = 2,
    eval_max_workers: int = 2,
):
    """
    Evaluate the search and generation capabilities of the RAG system.

    Args:
        rag (BaseRAG): The RAG system to evaluate.
        eval_file_path (str): Path to the evaluation file.
        output_file (str, optional): Path to save the evaluation results. Defaults to `None`.
        eval_llm (BaseCompletionLLM, optional): LLM to use for evaluate llm as a judge. Defaults to `None`.

    """
    logger.info("RAG metadata:")
    logger.info(rag.get_metadata())

    logger.info("Ingesting evaluation data...")

    rag.ingest_db(file_or_folder_paths=[eval_file_path], num_workers=1)

    try:
        logger.info(f"Loading: {eval_file_path}")
        with open(eval_file_path, "r") as f:
            data = json.load(f)

        eval_data = data["data"]
        context_id_to_context = {
            context["context_id"]: context["context"] for context in data["contexts"]
        }

        logger.info(f"Total contexts: {len(data['contexts'])}")
        logger.info(f"Total questions: {len(eval_data)}")

        answers = []
        referenced_answers = []
        contexts = []
        queries = []

        correct_count = 0
        sum_rr = 0
        count = 0
        retrieved_time = 0
        generation_time = 0

        bar = tqdm(total=len(eval_data), desc="Evaluating ...")
        for item in eval_data:
            count += 1
            question = item["question"]
            true_idx = item["context_id"]

            response = rag.search(query=question, return_retrieved_result=True, top_k=5)

            retrieved_time += response.retrieved_time
            generation_time += response.generation_time
            retrieved_idx = [
                doc.metadata["context_id"] for doc in response.retrieved_results
            ]

            rr = calculate_rr(true_idx=true_idx, retrieved_idx=retrieved_idx)
            sum_rr += rr

            answers.append(response.content)
            contexts.append(context_id_to_context[true_idx])
            queries.append(question)
            referenced_answers.append(item["answer"])

            bar.update(1)
            bar.set_postfix(
                {
                    "current_mrr": sum_rr / count,
                    "avg_retrieved_time": retrieved_time / count,
                    "avg_generation_time": generation_time / count,
                }
            )

        bar.close()
        accuracy = correct_count / len(eval_data)
        mrr = sum_rr / len(eval_data)

        logger.info("Evaluating answers...")
        judged_responses = llm_as_a_judge_inference(
            llm=eval_llm if eval_llm else rag.get_llm(),
            queries=queries,
            contexts=contexts,
            referenced_answers=referenced_answers,
            answers=answers,
            batch_size=eval_batch_size,
            max_workers=eval_max_workers,
        )

        accuracy = sum(
            [1 if response.approved else 0 for response in judged_responses]
        ) / len(judged_responses)

        results = {
            "accuracy": accuracy,
            "mrr": mrr,
            "avg_retrieved_time": retrieved_time / len(eval_data),
            "avg_generation_time": generation_time / len(eval_data),
        }

        display_results(
            title=data["dataset_name"],
            headers=["Metric", "Value"],
            results=results,
        )

        if output_file:
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "dataset_name": data["dataset_name"],
                        "metadata": rag.get_metadata(),
                        **results,
                    },
                    f,
                )
            logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise e

    finally:
        logger.info("Cleaning up resources...")
        rag.clean_resource()
