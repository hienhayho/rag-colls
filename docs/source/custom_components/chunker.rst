Custom chunker
==================

Please follow these instructions to create a custom chunker.

**1:** Place your custom chunker `.py` in `rag_colls/processors/chunkers` directory.

Let's say you want to create a custom chunker called `MyChunker`, you would create a file named `my_chunker.py` in the `rag_colls/processors/chunkers` directory.

-   The file structure should look like this:

    .. code-block:: text

        rag_colls/
        ├── processors/
        │   ├── chunkers/
        │   │   ├── my_chunker.py
        │   │   └── ...
        │   └── ...
        └── ...

**2:** Implement your custom chunker.

-   Your custom chunker must inherit from the `BaseChunker` class. Here is the code for `BaseChunker`:

    .. code-block:: python

        from abc import ABC, abstractmethod
        from rag_colls.types.core.document import Document


        class BaseChunker(ABC):
            @abstractmethod
            def _chunk(self, documents: list[Document], **kwargs) -> list[Document]:
                """
                Chunk the documents.

                Args:
                    documents (list[Document]): List of documents to be chunked.
                    **kwargs: Additional keyword arguments for the chunking function.

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
                    **kwargs: Additional keyword arguments for the chunking function.

                Returns:
                    list[Document]: List of chunked documents.
                """
                raise NotImplementedError("This method should be overridden by subclasses.")

            ...

You must implement `_chunk` method, `_achunk` is optional since it will call `_chunk` method asynchronously if not implemented.

- Example of `MyChunker`:

    .. code-block:: python

        from rag_colls.processors.chunkers import BaseChunker
        from rag_colls.types.core.document import Document

        class MyChunker(BaseChunker):
            def _chunk(self, documents: list[Document], **kwargs) -> list[Document]:
                # Implement your chunking logic here
                chunked_documents = []
                for doc in documents:
                    # Example chunking logic
                    chunks = [doc.text[i:i + 100] for i in range(0, len(doc.text), 100)]
                    for chunk in chunks:
                        chunked_documents.append(Document(text=chunk))
                return chunked_documents

**3:** Usage

-   You can use your custom chunker in the same way as the built-in chunkers.
    .. code-block:: python
        from rag_colls.types.core.document import Document
        from rag_colls.processors.chunkers.my_chunker import MyChunker

        chunker = MyChunker()
        documents = [Document(content="This is a long document that needs to be chunked.")]
        chunked_documents = chunker.chunk(documents)

        print(chunked_documents)

- Or use it in initialize RAG instance:

    .. code-block:: python
        from rag_colls.rags.basic_rag import BasicRAG
        from rag_colls.processors.chunkers.my_chunker import MyChunker

        rag = BasicRAG(
            ...,
            chunker=MyChunker(),
            ...
        )
        ...
