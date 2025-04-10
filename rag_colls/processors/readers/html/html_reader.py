from pathlib import Path
from typing import Optional

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader


class HTMLReader(BaseReader):
    """Reader HTML usimg html2text

    Reader behavior:
        - HTML is read with html2text.
        - All of the texts will be split by `page_break_pattern`
        - Each page is extracted as a Document
        - The output is a list of Documents

    Args:
        page_break_pattern (str): Pattern to split the HTML into pages
    """

    def __init__(self, page_break_pattern: Optional[str] = None, *args, **kwargs):
        self._page_break_pattern: Optional[str] = page_break_pattern
        super().__init__()

    def _load_data(
        self,
        file_path: Path | str,
        should_split: bool = True,
        extra_info: Optional[dict] = None,
        encoding: str = "utf-8",
    ) -> list[Document]:
        """Load data using Html reader

        Args:
            file_path: path to HTML file
            should_split: whether to split the HTML into pages
            extra_info: extra information passed to this reader during extracting data

        Returns:
            list[Document]: list of documents extracted from the HTML file
        """
        try:
            import html2text  # noqa
        except ImportError:
            raise ImportError(
                "html2text is not installed. "
                "Please install it using `pip install html2text`"
            )

        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with file_path.open("r", encoding=encoding) as f:
            html_text = "".join([line[:-1] for line in f.readlines()])

        # read HTML
        all_text = html2text.html2text(html_text)
        pages = (
            all_text.split(self._page_break_pattern)
            if self._page_break_pattern
            else [all_text]
        )

        extra_info = extra_info or {
            "file_path": str(file_path),
            "should_split": should_split,
            "file_size": file_path.stat().st_size,
            "file_type": "html",
            "encoding": encoding,
        }

        documents = []
        for page_id, page in enumerate(pages):
            page_extra_info = extra_info.copy()
            page_extra_info["source"] = f"{file_path.name}: Page {page_id + 1}"
            documents.append(
                Document(
                    document=page.strip().encode(encoding), metadata=page_extra_info
                )
            )

        return documents
