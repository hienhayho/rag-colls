import argparse
from rag_colls.llms.litellm_llm import LiteLLM
from rag_colls.databases.bm25.bm25s import BM25s
from rag_colls.embeddings.openai_embedding import OpenAIEmbedding
from rag_colls.rerankers.weighted_reranker import WeightedReranker
from rag_colls.processors.chunkers.semantic_chunker import SemanticChunker
from rag_colls.databases.vector_databases.chromadb import ChromaVectorDatabase

from rag_colls.rags.raft import RAFT, CONTEXTUAL_BOOST_SYSTEM_PROMPT

# from rag_colls.core.utils import load_chunks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system.",
    )
    parser.add_argument(
        "--f",
        nargs="+",
        type=str,
        help="File paths or folders to ingest",
        default=["samples/data/2503.20376v1.pdf"],
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for the ingested data",
        default="test.jsonl",
    )
    return parser.parse_args()


def get_rag():
    return RAFT(
        vector_database=ChromaVectorDatabase(
            persistent_directory="./chroma_db", collection_name="test"
        ),
        bm25=BM25s(
            save_dir="./bm25s",
        ),
        reranker=WeightedReranker(weights=[0.8, 0.2]),  # [semantic_weight, bm25_weight]
        chunker=SemanticChunker(embed_model_name="text-embedding-ada-002"),
        llm=LiteLLM(model_name="openai/gpt-4o-mini"),
        embed_model=OpenAIEmbedding(model_name="text-embedding-ada-002"),
        gen_contextual_system_prompt=CONTEXTUAL_BOOST_SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    args = parse_args()
    rag = get_rag()

    rag.ingest_db(
        file_or_folder_paths=args.f,
        batch_embedding=100,
        save_chunk=True,
        output_path=args.output_path,
    )
