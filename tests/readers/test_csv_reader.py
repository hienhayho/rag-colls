from rag_colls.processors.readers.csv import CSVReader


def test_csv_reader():
    """
    Test the CSVReader class.
    """
    reader = CSVReader()

    documents = reader.load_data(file_path="samples/data/test.csv")

    assert len(documents) > 0, "No documents found in the CSV file."

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
    assert metadata["file_type"] == "csv", "Incorrect file type in metadata"
    assert "file_size" in metadata, "Metadata missing file_size"
    assert "num_rows" in metadata, "Metadata missing num_rows"
    assert "num_cols" in metadata, "Metadata missing num_cols"
    assert "columns" in metadata, "Metadata missing columns"
    assert "source" in metadata, "Metadata missing source"

    primitives = (bool, str, int, float, type(None), list)
    for _, value in metadata.items():
        assert isinstance(value, primitives), (
            "Metadata values should be bool, str, int, float, or None"
        )
