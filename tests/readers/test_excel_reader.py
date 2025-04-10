from rag_colls.processors.readers.excel import ExcelReader


def test_excel_reader():
    """
    Test the ExcelReader class.
    """
    reader = ExcelReader()

    documents = reader.load_data(file_path="samples/data/test.xlsx")

    assert len(documents) > 0, "No documents found in the Excel file."

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
    assert "source" in metadata, "Metadata missing source"
    assert metadata["file_type"] == "excel", "Incorrect file type in metadata"
    assert "num_rows" in metadata, "Metadata missing num_rows"
    assert "num_cols" in metadata, "Metadata missing num_cols"
    assert "columns" in metadata, "Metadata missing columns"

    primitives = (bool, str, int, float, type(None), list)
    for _, value in metadata.items():
        assert isinstance(value, primitives), (
            "Metadata values should be bool, str, int, float, or None"
        )
