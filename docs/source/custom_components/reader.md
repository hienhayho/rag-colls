# Custom Reader

Please follow these instructions to create a custom reader.

## Create folder structure for your custom reader

Place your custom reader `.py` file in the `rag_colls/processors/readers` directory.

Let's say you want to create a custom reader called `MyReader` which is used to read `<ext>` files.

You would create a file named `my_reader.py` in the `rag_colls/processors/readers/<ext>/` directory.

The file structure should look like this:

```text
rag_colls/
├── processors/
│   ├── readers/
│   │   ├── <ext>/
│   │   │   ├── __init__.py
│   │   │   ├── my_reader.py
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

For example: `PyMuPDFReader` is used to read `.pdf` files, so the file structure would look like this:

```text
rag_colls/
├── processors/
│   ├── readers/
│   │   ├── pdf/
│   │   │   ├── __init__.py
│   │   │   ├── pymupdf_reader.py
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

## Implement your custom reader

In your custom reader file, you need to create a class that inherits from the `BaseReader` class.

Here's the code for `BaseReader`:

```python
from abc import ABC, abstractmethod
from rag_colls.types.core.document import Document

class BaseReader(ABC):
    @abstractmethod
    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Loads data from the specified file path and returns a list of Document objects.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            should_split (bool): Whether to split the data into smaller chunks.
            extra_info (dict | None): Additional information to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
```

> **Note**
> You must add `should_split` and `extra_info` into `metadata` of each `Document` object.

### Example: `MyCustomTxtReader`

Here is an example of a custom reader that reads .txt files and splits the content into smaller chunks.

First, create a directory for your custom reader if it doesn't exist. The directory structure should look like this:

```text
rag_colls/
├── processors/
│   ├── readers/
│   │   ├── txt/
│   │   │   ├── __init__.py
│   │   │   ├── my_custom_txt_reader.py
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

Then, create a file named `my_custom_txt_reader.py` in the `rag_colls/processors/readers/txt/` directory.
In this file, you can implement your custom reader class like this:

```python
from pathlib import Path
from rag_colls.core.base.readers.base import BaseReader
from rag_colls.types.core.document import Document

class MyCustomTxtReader(BaseReader):
    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Loads data from .txt file and return its documents.

        Args:
            file_path (str | Path): The path to the file to be loaded.
            should_split (bool): Whether to split the data into smaller chunks.
            extra_info (dict | None): Additional information to be passed to the loader.

        Returns:
            list[Document]: A list of Document objects.
        """
        documents = []
        with open(file_path, "r") as file:
            content = file.read()
            if should_split:
                chunks = content.split("\n\n")  # Example: split by double newlines
                for chunk in chunks:
                    documents.append(Document(
                        document=chunk,
                        metadata={"should_split": should_split, **(extra_info or {})}
                    ))
            else:
                documents.append(Document(
                    document=content,
                    metadata={"should_split": should_split, **(extra_info or {})}
                ))
        return documents
```

Finally, add it in `rag_colls/processors/readers/txt/__init__.py` file:

```python
...
from .my_custom_txt_reader import MyCustomTxtReader

__all__ = [..., "MyCustomTxtReader"]
```

## Usage

You can use your custom reader in the same way as the built-in readers.

```python
from rag_colls.processors.readers.txt import MyCustomTxtReader

reader = MyCustomTxtReader()
documents = reader.load_data(file_path="path/to/your/file.txt")

for doc in documents:
    print(doc.document)
    print(doc.metadata)
```

## Create a test for your custom reader

Remember to create test case for your custom reader. You can refer to `tests/readers/test_pdf_reader.py` for more information.

In `tests/readers` directory, create a file named `test_txt_reader.py` and implement your test case.

```python
from rag_colls.processors.readers.txt import MyCustomTxtReader

def test_custom_txt_reader():
    reader = MyCustomTxtReader()
    documents = reader.load_data(file_path="samples/data/test.txt")

    assert len(documents) > 0, "No documents found"

    first_document = documents[0]
    assert hasattr(first_document, "document"), "Missing `document` attribute"
    assert hasattr(first_document, "metadata"), "Missing `metadata` attribute"
```

## Register as default reader (Optional)

In case you want to add your custom reader to the default readers list, you can do so by modifying the `rag_colls/processors/file_processor.py` file.

Find the `_get_default_processors` method in the `FileProcessor` class and add your custom reader to it.

```python
class FileProcessor:
    ...
    def _get_default_processors(self) -> dict[str, BaseReader]:
        """
        Initialize default file processors.

        Returns:
            dict[str, BaseReader]: A dictionary of default file processors.
        """
        from .readers.txt import MyCustomTxtReader

        return {
            ...
            ".txt": MyCustomTxtReader(),
            ...
        }
```

## Add to the documentation (Optional)

_Update later._
