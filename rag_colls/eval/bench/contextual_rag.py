from rag_colls.llms.vllm_llm import VLLM
from rag_colls.llms.litellm_llm import LiteLLM
from rag_colls.databases.bm25.bm25s import BM25s
from rag_colls.embeddings.hf_embedding import HuggingFaceEmbedding
from rag_colls.rerankers.weighted_reranker import WeightedReranker
from rag_colls.processors.chunkers.semantic_chunker import SemanticChunker
from rag_colls.databases.vector_databases.chromadb import ChromaVectorDatabase

from rag_colls.rags.contextual_rag import ContextualRAG, CONTEXTUAL_PROMPT

from rag_colls.eval.source.eval_reader import eval_file_processor
from rag_colls.eval.source.eval import eval_search_and_generation, get_eval_args

if __name__ == "__main__":
    args = get_eval_args()

    rag = ContextualRAG(
        vector_database=ChromaVectorDatabase(
            persistent_directory="./chroma_db", collection_name="test"
        ),
        bm25=BM25s(save_dir="./bm25s"),
        reranker=WeightedReranker(weights=[0.8, 0.2]),  # [semantic_weight, bm25_weight]
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
        gen_contextual_prompt_template=CONTEXTUAL_PROMPT,
    )

    eval_search_and_generation(
        rag=rag,
        eval_file_path=args.f,
        output_file=args.o,
        eval_llm=LiteLLM(model_name="openai/gpt-4o-mini"),
        eval_batch_size=args.eval_batch_size,
        eval_max_workers=args.eval_max_workers,
    )
