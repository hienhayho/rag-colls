from rag_colls.processors.file_processor import FileProcessor
from pathlib import Path


def test_file_processor():
    file_processor = FileProcessor()
    file_paths = [str(file) for file in Path("samples/data").glob("*")]
    documents = file_processor.load_data(
        file_or_folder_paths=file_paths,
        should_splits=[True] * len(file_paths),
        extra_infos=[None] * len(file_paths),
    )
    assert len(documents) > 0, "No documents found in the data directory."
    print("len(file_paths):", len(file_paths))
    print("len(documents):", len(documents))
    for i in range(len(documents)):
        print("documents[{}].metadata:".format(i), documents[i].metadata)


def test_file_processor_dir():
    file_processor = FileProcessor()
    file_paths = ["samples"]
    documents = file_processor.load_data(
        file_or_folder_paths=file_paths,
    )
    assert len(documents) > 0, "No documents found in the data directory."
    print("len(documents):", len(documents))
    for i in range(len(documents)):
        print("documents[{}].metadata:".format(i), documents[i].metadata)
