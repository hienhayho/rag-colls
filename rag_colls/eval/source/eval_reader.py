from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader

from rag_colls.processors.file_processor import FileProcessor


class EvalJsonReader(BaseReader):
    def _load_data(
        self, file_path: str, should_split: bool, extra_info: dict, **kwargs
    ):
        """
        Load data from a JSON test file.
        """
        import json

        with open(file_path, "r") as f:
            data = json.load(f)

        documents: list[Document] = []
        for item in data["data"]:
            doc = Document(
                document=item["context"],
                metadata={
                    "should_split": False,
                    "context_id": item["context_id"],
                    "answer": item["answer"],
                },
            )
            documents.append(doc)

        return documents


eval_file_processor = FileProcessor(
    processors={".json": EvalJsonReader()}, merge_with_default_processors=False
)
