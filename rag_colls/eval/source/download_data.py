import json
from tqdm import tqdm
from uuid import uuid4
from datasets import load_dataset


def load_squad_v2():
    dataset = load_dataset(
        "rajpurkar/squad_v2", split="validation", cache_dir="data_cache"
    )

    results = {
        "dataset_name": "squad_v2",
        "metadata": {
            "description": "SQuAD v2.0 dataset for question answering.",
            "url": "https://huggingface.co/datasets/rajpurkar/squad_v2/",
        },
        "contexts": [],
        "data": [],
    }

    previous_ctx_id = None
    previous_ctx = ""

    for i in tqdm(range(len(dataset)), desc="Loading SQuAD v2.0 ..."):
        if i == 0:
            context_id = str(uuid4())
            results["contexts"].append(
                {
                    "context_id": context_id,
                    "context": dataset[i]["context"],
                }
            )
            results["data"].append(
                {
                    "question_id": str(uuid4()),
                    "context_id": context_id,
                    "question": dataset[i]["question"],
                    "answer": (
                        dataset[i]["answers"]["text"][0]
                        if len(dataset[i]["answers"]["text"]) > 0
                        else ""
                    ),
                }
            )
            previous_ctx_id = results["data"][0]["context_id"]
            previous_ctx = dataset[i]["context"]
        else:
            if dataset[i]["context"] != previous_ctx:
                context_id = str(uuid4())
                results["contexts"].append(
                    {
                        "context_id": context_id,
                        "context": dataset[i]["context"],
                    }
                )
                results["data"].append(
                    {
                        "question_id": str(uuid4()),
                        "context_id": context_id,
                        "question": dataset[i]["question"],
                        "answer": (
                            dataset[i]["answers"]["text"][0]
                            if len(dataset[i]["answers"]["text"]) > 0
                            else ""
                        ),
                    }
                )
                previous_ctx_id = results["data"][-1]["context_id"]
                previous_ctx = dataset[i]["context"]
            else:
                results["data"].append(
                    {
                        "question_id": str(uuid4()),
                        "context_id": previous_ctx_id,
                        "question": dataset[i]["question"],
                        "answer": (
                            dataset[i]["answers"]["text"][0]
                            if len(dataset[i]["answers"]["text"]) > 0
                            else ""
                        ),
                    }
                )

    print(f"Total number of contexts: {len(results['contexts'])}")
    print(f"Total number of questions: {len(results['data'])}")

    with open("data/squad_v2_validation.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    load_squad_v2()
