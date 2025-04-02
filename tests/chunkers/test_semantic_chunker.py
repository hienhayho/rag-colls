from rag_colls.processors.file_processor import FileProcessor

processor = FileProcessor()

documents = processor.load_data(file_paths=["samples/data/2503.20376v1.pdf"])
not_be_chunked_documents = processor.load_data(
    file_paths=["samples/data/2503.20376v1.pdf"],
    should_splits=[False],
)


def test_semantic_chunker():
    from rag_colls.processors.chunkers.semantic_chunker import SemanticChunker

    chunker = SemanticChunker()

    chunked_documents = chunker.chunk(documents)

    assert len(chunked_documents) > len(documents), (
        "Chunked documents should be more than original documents."
    )

    assert len(chunked_documents) == len(not_be_chunked_documents), (
        "Chunked documents should be same as original documents."
    )

    first_chunk = chunked_documents[0]

    assert hasattr(first_chunk, "document"), "Chunk does not have document attribute."
    assert hasattr(first_chunk, "metadata"), "Chunk does not have metadata attribute."
