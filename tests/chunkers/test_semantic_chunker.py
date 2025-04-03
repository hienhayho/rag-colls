from rag_colls.processors.file_processor import FileProcessor

processor = FileProcessor()

documents = processor.load_data(file_or_folder_paths=["samples/data/2503.20376v1.pdf"])
not_be_chunked_documents = processor.load_data(
    file_or_folder_paths=["samples/data/2503.20376v1.pdf"],
    should_splits=[False],
)


def test_semantic_chunker():
    from rag_colls.processors.chunkers.semantic_chunker import SemanticChunker

    chunker = SemanticChunker(mocking=True)

    chunked_documents = chunker.chunk(documents)

    assert len(chunked_documents) >= len(documents), (
        "Chunked documents should be more than original documents."
    )

    not_be_chunked_documents_chunked = chunker.chunk(not_be_chunked_documents)

    assert len(not_be_chunked_documents_chunked) == len(not_be_chunked_documents), (
        "Chunked with should_splits False should be equal to original documents."
    )

    first_chunk = chunked_documents[0]
    assert hasattr(first_chunk, "document"), "Chunk does not have document attribute."
    assert hasattr(first_chunk, "metadata"), "Chunk does not have metadata attribute."

    first_not_be_chunked_document = not_be_chunked_documents_chunked[0]
    assert hasattr(first_not_be_chunked_document, "document"), (
        "Chunk does not have document attribute."
    )

    assert hasattr(first_not_be_chunked_document, "metadata"), (
        "Chunk does not have metadata attribute."
    )
