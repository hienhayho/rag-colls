from rag_colls.embeddings.hf_embedding import HuggingFaceEmbedding
from rag_colls.types.core.document import Document


def test_hf_embedding():
    """
    Test the HuggingFaceEmbedding class.
    """
    embedding = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    sample = embedding._get_query_embedding("Hello, world!")
    assert sample.embedding is not None
    assert len(sample.embedding) == 768, "Embedding dimensions should be 768"

    sample = embedding._get_document_embedding(Document(document="Hello, world!"))
    assert sample.embedding is not None
    assert len(sample.embedding) == 768, "Embedding dimensions should be 768"

    samples = embedding._get_batch_query_embedding(["Hello, world!", "Hello, world!"])
    assert len(samples) == 2
    assert len(samples[0].embedding) == 768, "Embedding dimensions should be 768"

    samples = embedding._get_batch_document_embedding(
        [Document(document="Hello, world!"), Document(document="Hello, world!")]
    )
    assert len(samples) == 2
    assert len(samples[0].embedding) == 768, "Embedding dimensions should be 768"
