Custom Chunker
==============

Please follow these instructions to create a custom chunker.

**1.** Place your custom chunker **.py** file in the **rag_colls/processors/chunkers** directory.

Let's say you want to create a custom chunker called **MyChunker**, you would create a file named **my_chunker.py** in the **rag_colls/processors/chunkers** directory.

The file structure should look like this:

.. code-block:: text

    rag_colls/
    ├── processors/
    │   ├── chunkers/
    │   │   ├── my_chunker.py
    │   │   └── ...
    │   └── ...
    └── ...

**2.** Implement your custom chunker.

Your custom chunker must inherit from the **BaseChunker** class. Here's the code for **BaseChunker**:

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

You must implement **_chunk** and **_achunk**. With **_achunk** method, you can call **_chunk** asynchronously using **asyncio.to_thread**.

**Example: MyChunker**

.. code-block:: python

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
            self, documents: list[Document], show_progress: bool = False, **kwargs
        ):
            return await asyncio.to_thread(self._chunk, documents, **kwargs)

**3.** Usage

You can use your custom chunker like any built-in chunker:

.. code-block:: python

    from rag_colls.types.core.document import Document
    from rag_colls.processors.chunkers.my_chunker import MyChunker

    chunker = MyChunker()
    documents = [Document(document="This is a long document that needs to be chunked.")]
    chunked_documents = chunker.chunk(documents)

    print(chunked_documents)

Or use it while initializing a RAG instance:

.. code-block:: python

    from rag_colls.rags.basic_rag import BasicRAG
    from rag_colls.processors.chunkers.my_chunker import MyChunker

    rag = BasicRAG(
        ...,
        chunker=MyChunker(),
        ...
    )
