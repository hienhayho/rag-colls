from rag_colls.llms.vllm_llm import VLLM
from rag_colls.databases.bm25.bm25s import BM25s
from rag_colls.embeddings.openai_embedding import OpenAIEmbedding
from rag_colls.rerankers.weighted_reranker import WeightedReranker
from rag_colls.processors.chunkers.semantic_chunker import SemanticChunker
from rag_colls.databases.vector_databases.chromadb import ChromaVectorDatabase

from rag_colls.rags.raft import RAFT, CONTEXTUAL_BOOST_SYSTEM_PROMPT, PromptModeEnum

rag = RAFT(
    vector_database=ChromaVectorDatabase(
        persistent_directory="./chroma_db", collection_name="test"
    ),
    bm25=BM25s(
        save_dir="./bm25s",
    ),
    reranker=WeightedReranker(weights=[0.8, 0.2]),  # [semantic_weight, bm25_weight]
    chunker=SemanticChunker(embed_model_name="text-embedding-ada-002"),
    llm=VLLM(model_name="<your-trained-model-path>"),
    embed_model=OpenAIEmbedding(model_name="text-embedding-ada-002"),
    gen_contextual_system_prompt=CONTEXTUAL_BOOST_SYSTEM_PROMPT,
)

response = rag.search(
    query="What is text embedding ?", top_k=5, prompt_mode=PromptModeEnum.JSON
)

print(response.content)
print("===========")
print(response.usage)
