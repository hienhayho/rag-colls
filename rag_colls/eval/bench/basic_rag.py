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
    "--o",
    type=str,
    help="Path to save the evaluation results",
)
args = parser.parse_args()

rag = BasicRAG(
    vector_database=ChromaVectorDatabase(
        persistent_directory="./chroma_db", collection_name="benchmark"
    ),
    processor=eval_file_processor,
    chunker=SemanticChunker(embed_model_name="text-embedding-ada-002"),
    llm=VLLM(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.6,
        dtype="half",
        download_dir="./model_cache",
    ),
    embed_model=HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        cache_folder="./model_cache",
        device="cuda:1",
    ),
)

eval_search_and_generation(
    rag=rag,
    eval_file_path=args.f,
    output_file=args.o,
    eval_llm=LiteLLM(model_name="openai/gpt-4o-mini"),
)
