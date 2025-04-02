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

    assert len(chunker.chunk(documents)) >= len(documents)

    assert len(chunker.chunk(not_be_chunked_documents)) == len(not_be_chunked_documents)
