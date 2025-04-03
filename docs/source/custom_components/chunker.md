# Custom Chunker

Please follow these instructions to create a custom chunker.

## Create folder structure for your custom chunker

Place your custom chunker `.py` file in the `rag_colls/processors/chunkers` directory.

Let's say you want to create a custom chunker called `MyChunker`, you would create a file named `my_chunker.py` in the `rag_colls/processors/chunkers` directory.

The file structure should look like this:

```text
rag_colls/
├── processors/
│   ├── chunkers/
│   │   ├── my_chunker.py
│   │   └── ...
│   └── ...
└── ...
```

## Implement your custom chunker

Your custom chunker must inherit from the `BaseChunker` class. Here's the code for `BaseChunker`:

```python
from abc import ABC, abstractmethod
from rag_colls.types.core.document import Document

class BaseChunker(ABC):
    @abstractmethod
    def _chunk(self, documents: list[Document], **kwargs) -> list[Document]:
        """
        Chunk the documents.

        Args:
            documents (list[Document]): List of documents to be chunked.
            `kwargs: Additional keyword arguments for the chunking function.

        Returns:
            list[Document]: List of chunked documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def _achunk(self, documents: list[Document], **kwargs) -> list[Document]:
        """
        Asynchronously chunk the documents.

        Args:
            documents (list[Document]): List of documents to be chunked.
            `kwargs: Additional keyword arguments for the chunking function.

        Returns:
            list[Document]: List of chunked documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
```

You must implement `_chunk` and `_achunk`. With `_achunk` method, you can call `_chunk` asynchronously using `asyncio.to_thread`.

### Example: `MyChunker`

```python
import asyncio
from rag_colls.core.base.chunkers.base import BaseChunker
from rag_colls.types.core.document import Document

class MyChunker(BaseChunker):
    def _chunk(self, documents: list[Document], **kwargs) -> list[Document]:
        # Implement your chunking logic here
        chunked_documents = []
        for doc in documents:
            # Example chunking logic
            chunks = [doc.document[i:i + 100] for i in range(0, len(doc.document), 100)]
            for chunk in chunks:
                chunked_documents.append(Document(document=chunk, metadata=doc.metadata))
        return chunked_documents

    async def _achunk(
        self, documents: list[Document], **kwargs
    ):
        return await asyncio.to_thread(self._chunk, documents, **kwargs)
```

## Usage

You can use your custom chunker like any built-in chunker:

```python
from rag_colls.types.core.document import Document
from rag_colls.processors.chunkers.my_chunker import MyChunker

chunker = MyChunker()
documents = [Document(document="This is a long document that needs to be chunked.")]
chunked_documents = chunker.chunk(documents)

print(chunked_documents)
```

Or use it while initializing a RAG instance:

```python
from rag_colls.rags.basic_rag import BasicRAG
from rag_colls.processors.chunkers.my_chunker import MyChunker

rag = BasicRAG(
    ...,
    chunker=MyChunker(),
    ...
)
```

## Create a test for your custom chunker

Remember to create test case for your custom chunker. You can refer to `tests/chunkers/test_semantic_chunker.py` for more information.

In `tests/chunkers` directory, create a file named `test_my_chunker.py` and implement your test case.

```python
from rag_colls.types.core.document import Document

def test_my_chunker():
    """
    Test the custom chunker.
    """
    from rag_colls.processors.chunkers.my_chunker import MyChunker

    chunker = MyChunker()

    documents = [Document(document="This is a long document that needs to be chunked.")]

    chunked_documents = chunker.chunk(documents)

    assert len(chunked_documents) > 0, "No chunked documents found"

    first_chunk = chunked_documents[0]

    assert hasattr(first_chunk, "document"), "Chunk does not have document attribute."
    assert hasattr(first_chunk, "metadata"), "Chunk does not have metadata attribute."
```

## Add to the documentation (Optional)

_Update later._
