from elasticsearch import Elasticsearch
from rag_colls.llms.litellm_llm import LiteLLM
from rag_colls.databases.bm25.elastic_search import ElasticSearch
from rag_colls.embeddings.hf_embedding import HuggingFaceEmbedding
from rag_colls.rerankers.weighted_reranker import WeightedReranker
from rag_colls.processors.chunkers.semantic_chunker import SemanticChunker
from rag_colls.databases.vector_databases.chromadb import ChromaVectorDatabase

from rag_colls.rags.contextual_rag import ContextualRAG, CONTEXTUAL_PROMPT
from rag_colls.llms.vllm_llm import VLLM

# llm = VLLM(
#     model_name="Qwen/Qwen2.5-3B-Instruct",
#     gpu_memory_utilization=0.5,
#     dtype="half",
#     download_dir="./model_cache",
# )

def ingest_es():
    es = Elasticsearch(
        "http://es_os:9200",
    )
    if not es.indices.exists(index="documents_bm25"):
        es.indices.create(index="documents_bm25")

    print(es.indices.get_mapping(index="documents_bm25"))
# ingest_es()
rag = ContextualRAG(
    vector_database=ChromaVectorDatabase(
        persistent_directory="./chroma_db", collection_name="test"
    ),
    bm25=ElasticSearch(
        host="http://es_os:9200",
        index_name="documents_bm25",
    ),
    reranker=WeightedReranker(weights=[0.8, 0.2]),  # [semantic_weight, bm25_weight]
    chunker=SemanticChunker(embed_model_name="BAAI/bge-base-en-v1.5"),
    llm=LiteLLM(model_name="openai/gpt-4o-mini"),
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5"),
    gen_contextual_prompt_template=CONTEXTUAL_PROMPT,
)

rag.ingest_db(
    file_or_folder_paths=["samples/papers/2409.13588v1.pdf"], batch_embedding=100
)
