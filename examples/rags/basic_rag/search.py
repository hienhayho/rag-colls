from rag_colls.rags.basic_rag import BasicRAG
from rag_colls.llms.litellm_llm import LiteLLM
from rag_colls.embeddings.openai_embedding import OpenAIEmbedding
from rag_colls.processors.chunkers.semantic_chunker import SemanticChunker
from rag_colls.databases.vector_databases.chromadb import ChromaVectorDatabase

rag = BasicRAG(
    vector_database=ChromaVectorDatabase(
        persistent_directory="./chroma_db", collection_name="test"
    ),
    chunker=SemanticChunker(embed_model_name="text-embedding-ada-002"),
    llm=LiteLLM(model_name="openai/gpt-4o-mini"),
    embed_model=OpenAIEmbedding(model_name="text-embedding-ada-002"),
)

response = rag.search(query="What is Chain Buddy ?", top_k=5)

print(response.content)
print("===========")
print(response.usage)
