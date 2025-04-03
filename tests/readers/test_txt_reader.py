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
