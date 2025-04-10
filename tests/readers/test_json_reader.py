from rag_colls.processors.readers.json import JSONReader


def test_json_reader():
    """
    Test the JSONReader class.
    """
    reader = JSONReader()

    documents = reader.load_data(file_path="samples/data/test.json")

    assert len(documents) > 0, "No documents found in the JSON file."

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
    assert metadata["file_type"] == "json", "Incorrect file type in metadata"
    assert "file_size" in metadata, "Metadata missing file_size"
    assert "encoding" in metadata, "Metadata missing encoding"
    assert "source" in metadata, "Metadata missing source"

    primitives = (bool, str, int, float, type(None))
    for _, value in metadata.items():
        assert isinstance(value, primitives), (
            "Metadata values should be bool, str, int, float, or None"
        )
