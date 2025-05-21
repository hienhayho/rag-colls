# RAFT: Retrieval-Augmented Fine-Tuning

[**RAFT (Retrieval-Augmented Fine-Tuning)**](https://techcommunity.microsoft.com/blog/aiplatformblog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/4084674) is a powerful fine-tuning framework that bridges the gap between retrieval-augmented generation (RAG) and supervised learning to enhance large language models (LLMs) in domain-specific question answering.

Developed by UC Berkeley and Microsoft researchers, RAFT trains LLMs to effectively reason over both relevant and irrelevant (distracting) context, significantly improving answer grounding and factual consistency—particularly in high-stakes domains like **healthcare**, **finance**, and **law**.

---

## 🚀 Quickstart Guide

### 1. 📥 Data Ingestion

Prepare your documents and convert them into chunked format using the `ingest.py` script:

```bash
python ingest.py --output-path test.jsonl
```

This will create a `test.jsonl` file containing all the processed chunks.

---

### 2. 🧾 Generate Fine-Tuning Dataset

Next, use the `gen_finetune_data.py` script to construct a supervised dataset for RAFT fine-tuning. You can choose your desired prompt format: `json`, `xml`, `plain_text`, `yaml`, or `markdown`.

```bash
python gen_finetune_data.py --f test.jsonl --output-path data.jsonl --prompt-mode json
```

---

### 3. 🧠 Fine-Tuning with LoRA

Install the fine-tuning dependencies:

```bash
pip install rag-colls[finetune]
```

Then run fine-tuning with a provided script, for example with `Qwen2.5-3B-Instruct`:

```bash
bash scripts/run_sft.sh Qwen/Qwen2.5-3B-Instruct data.jsonl qwen2_5_3b_outputs 1
```

For more fine-tuning options, check out the [ms-swift documentation](https://swift.readthedocs.io/en/latest/index.html).

---

### 4. 🔄 Merge LoRA and Export

After training, merge the LoRA adapter back into the base model using `export.sh`:

```bash
bash scripts/export.sh qwen2_5_3b_outputs/checkpoint-xxx qwen_2_5_3b_model
```

This will produce a fully merged model in `qwen_2_5_3b_model`.

---

### 5. 🔍 Inference with VLLM

Modify the model path in `search.py`:

```python
llm = VLLM(model_name="qwen_2_5_3b_model")
```

Then run the retrieval-based inference pipeline:

```bash
python search.py
```
