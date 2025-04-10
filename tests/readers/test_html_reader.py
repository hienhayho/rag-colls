from rag_colls.processors.readers.html import HTMLReader


def test_html_reader():
    """
    Test the HTMLReader class.
    """
    reader = HTMLReader()

    documents = reader.load_data(file_path="samples/data/test.html")

    assert len(documents) > 0, "No documents found in the HTML file."

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
    assert metadata["file_type"] == "html", "Incorrect file type in metadata"
    assert "file_size" in metadata, "Metadata missing file_size"
    assert "encoding" in metadata, "Metadata missing encoding"
    assert "source" in metadata, "Metadata missing source"

    primitives = (bool, str, int, float, type(None))
    for _, value in metadata.items():
        assert isinstance(value, primitives), (
            "Metadata values should be bool, str, int, float, or None"
        )
