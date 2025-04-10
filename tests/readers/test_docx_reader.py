from rag_colls.processors.readers.docx import DocxReader


def test_docx_reader():
    """
    Test the DOCXReader class.
    """
    reader = DocxReader()

    documents = reader.load_data(file_path="samples/data/test.docx")

    assert len(documents) > 0, "No documents found in the DOCX file."

    first_document = documents[0]
    assert hasattr(first_document, "document"), (
        "Document does not have document attribute."
    )
    assert hasattr(first_document, "metadata"), (
        "Document does not have metadata attribute."
    )

    # Test metadata
    metadata = first_document.metadata
    assert "file_path" in metadata, "Metadata missing file_path"
    assert "file_type" in metadata, "Metadata missing file_type"
    assert metadata["file_type"] == "docx", "Incorrect file type in metadata"
    assert "file_size" in metadata, "Metadata missing file_size"
    assert "source" in metadata, "Metadata missing source"

    primitives = (bool, str, int, float, type(None))
    for _, value in metadata.items():
        assert isinstance(value, primitives), (
            "Metadata values should be bool, str, int, float, or None"
        )
