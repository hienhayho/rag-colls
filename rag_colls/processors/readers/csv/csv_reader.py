# REF: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/tabular/base.py
from typing import Any, List, Optional, Dict
from pathlib import Path

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader


class CSVReader(BaseReader):
    """CSV parser.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

    """

    def __init__(self, *args: Any, concat_rows: bool = True, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows

    def _load_data(
        self,
        file: Path,
        should_split: bool = True,
        extra_info: Optional[Dict] = None,
        encoding: str = "utf-8",
    ) -> List[Document]:
        """Parse file.

        Returns:
            Union[str, List[str]]: a string or a List of strings.

        """
        try:
            import csv
        except ImportError:
            raise ImportError(
                "csv module is not installed. Please install it with 'pip install csv'."
            )

        if isinstance(file, str):
            file = Path(file)

        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")

        text_list = []
        with open(file, encoding=encoding) as fp:
            csv_reader = csv.reader(fp)
            for row in csv_reader:
                text_list.append(", ".join(row))

        metadata = {
            "filename": file.name,
            "extension": file.suffix,
            "source": f"{file.name}",
            "should_split": should_split,
            "file_size": file.stat().st_size,
            "file_path": str(file),
            "file_type": "csv",
            "encoding": encoding,
            "num_rows": len(text_list),
            "num_cols": len(text_list[0].split(",")),
            "columns": text_list[0].split(","),
        }

        if extra_info:
            metadata = {**metadata, **extra_info}

        if self._concat_rows:
            return [
                Document(
                    document="\n".join(text_list).encode(encoding), metadata=metadata
                )
            ]
        else:
            return [
                Document(document=text.encode(encoding), metadata=metadata)
                for text in text_list
            ]
