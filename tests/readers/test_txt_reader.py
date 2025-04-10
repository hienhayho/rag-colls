from rag_colls.processors.readers.txt import TxtReader


def test_txt_reader():
    """
    Test the TXTReader class.
    """
    reader = TxtReader()

    documents = reader.load_data(file_path="samples/data/test.csv")

    assert len(documents) > 0, "No documents found in the TXT file."

    first_document = documents[0]
    assert hasattr(first_document, "document"), (
        "Document does not have document attribute."
    )
    assert hasattr(first_document, "metadata"), (
        "Document does not have metadata attribute."
    )

    assert "source" in first_document.metadata, "Metadata missing source"

    metadata = first_document.metadata
    primitives = (bool, str, int, float, type(None))
    for _, value in metadata.items():
        assert isinstance(value, primitives), (
            "Metadata values should be bool, str, int, float, or None"
        )
