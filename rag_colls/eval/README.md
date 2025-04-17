# Eval system for rag-colls

## Dataset for evaluating search and generation process

Test file should be a `.json` file, the structure should be:

```json
{
    "dataset_name" : "...",
    "metadata": {...},
    "context": [
        {
            "context_id": uuid4(),
            "context": "...",
        },
        ...
    ],
    "data": [
        {
            "question_id": uuid4(),
            "context_id": uuid4(),
            "question": "...",
            "answer": "...", (Optional)
        },
        ...
    ]
}
```
