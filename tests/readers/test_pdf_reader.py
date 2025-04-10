from rag_colls.processors.readers.pdf import PyMuPDFReader


def test_pymupdf_reader():
    """
    Test the PyMuPDFReader class.
    """
    reader = PyMuPDFReader()

    documents = reader.load_data(file_path="samples/data/2503.20376v1.pdf")

    assert len(documents) > 0, "No documents found in the PDF file."

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
