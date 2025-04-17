import argparse

from rag_colls.llms.vllm_llm import VLLM
from rag_colls.llms.litellm_llm import LiteLLM
from rag_colls.rags.basic_rag import BasicRAG
from rag_colls.embeddings.hf_embedding import HuggingFaceEmbedding
from rag_colls.processors.chunkers.semantic_chunker import SemanticChunker
from rag_colls.databases.vector_databases.chromadb import ChromaVectorDatabase

from rag_colls.eval.source.eval_reader import eval_file_processor
from rag_colls.eval.source.eval import eval_search_and_generation

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

if __name__ == "__main__":
    args = parser.parse_args()
    rag = BasicRAG(
        vector_database=ChromaVectorDatabase(
            persistent_directory="./chroma_db", collection_name="benchmark"
        ),
        processor=eval_file_processor,
        chunker=SemanticChunker(embed_model_name="text-embedding-ada-002"),
        llm=VLLM(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="half",
            download_dir="./model_cache",
            tensor_parallel_size=args.num_gpus,
        ),
        embed_model=HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            cache_folder="./model_cache",
            device="cuda:4",
        ),
    )

    eval_search_and_generation(
        rag=rag,
        eval_file_path=args.f,
        output_file=args.o,
        eval_llm=LiteLLM(model_name="openai/gpt-4o-mini"),
        eval_batch_size=args.eval_batch_size,
        eval_max_workers=args.eval_max_workers,
    )
