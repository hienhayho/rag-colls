from rag_colls.rags.basic_rag import BasicRAG
from rag_colls.llms.litellm_llm import LiteLLM
from rag_colls.embeddings.hf_embedding import HuggingFaceEmbedding
from rag_colls.processors.chunkers.semantic_chunker import SemanticChunker
from rag_colls.databases.vector_databases.chromadb import ChromaVectorDatabase

rag = BasicRAG(
    vector_database=ChromaVectorDatabase(
        persistent_directory="./chroma_db", collection_name="test"
    ),
    chunker=SemanticChunker(embed_model_name="BAAI/bge-base-en-v1.5"),
    llm=LiteLLM(model_name="openai/gpt-4o-mini"),
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5"),
)

rag.ingest_db(file_or_folder_paths=["samples/data/"], batch_embedding=100)
