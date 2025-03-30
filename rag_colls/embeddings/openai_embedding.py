from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from rag_colls.types.embedding import Embedding
from rag_colls.types.core.document import Document
from rag_colls.core.base.embeddings.base import BaseEmbedding

load_dotenv()


class OpenAIEmbedding(BaseEmbedding):
    supported_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]
    defaul_model = "text-embedding-ada-002"

    client = OpenAI()

    def __init__(self, model_name: str | None = None):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name (str): The name of the OpenAI model to use. Can only be chosen from
                `[text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002]`.
        """
        if not model_name:
            model_name = self.defaul_model

        assert model_name in self.supported_models, (
            f"Model {model_name} is not supported."
        )

        self.model_name = model_name

    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Returns the embedding of the query.

        Args:
            query (str): The query to be embedded.

        Returns:
            Embedding: The embedding object of the query.
        """
        embedding = self.client.embeddings.create(model=self.model_name, input=query)
        return Embedding(
            embedding=embedding.data[0].embedding,
            metadata={
                "prompt_tokens": embedding.usage.prompt_tokens,
                "total_tokens": embedding.usage.total_tokens,
            },
        )

    def _get_document_embedding(self, document: Document) -> Embedding:
        """
        Returns the embedding of the document.
        Args:
            document (Document): The document to be embedded.

        Returns:
            Embedding: The embedding object of the document.
        """
        embedding = self.client.embeddings.create(
            model=self.model_name, input=document.document
        )
        return Embedding(
            embedding=embedding.data[0].embedding,
            metadata={
                "prompt_tokens": embedding.usage.prompt_tokens,
                "total_tokens": embedding.usage.total_tokens,
            },
        )

    def _get_batch_query_embedding(
        self, queries: list[str], **kwargs
    ) -> list[Embedding]:
        """
        Returns the embeddings of the queries.
        Args:
            queries (list[str]): The list of queries to be embedded.
            **kwargs: Additional keyword arguments for the embedding function.
        Returns:
            list[Embedding]: The list of embedding objects of the queries.
        """
        batch_size = kwargs.get("batch_size", 1)

        embeddings = []
        bar = tqdm(range(0, len(queries), batch_size), desc="Embedding ...")
        for i in bar:
            batch = queries[i : i + batch_size]
            embedding = self.client.embeddings.create(
                model=self.model_name, input=batch
            )
            for emb in embedding.data:
                embeddings.append(
                    Embedding(
                        embedding=emb.embedding,
                        metadata={
                            "prompt_tokens": embedding.usage.prompt_tokens,
                            "total_tokens": embedding.usage.total_tokens,
                        },
                    )
                )
            bar.update(n=len(batch))

        return embeddings

    def _get_batch_document_embedding(
        self, documents: list[Document], **kwargs
    ) -> list[Embedding]:
        """
        Returns the embeddings of the documents.
        Args:
            documents (list[Document]): The list of documents to be embedded.
            **kwargs: Additional keyword arguments for the embedding function.
        Returns:
            list[Embedding]: The list of embedding objects of the documents.
        """
        contents = [doc.document for doc in documents]

        batch_size = kwargs.get("batch_size", 1)
        embeddings = []

        bar = tqdm(range(0, len(contents), batch_size), desc="Embedding ...")
        for i in bar:
            batch = contents[i : i + batch_size]
            embedding = self.client.embeddings.create(
                model=self.model_name, input=batch
            )
            for emb in embedding.data:
                embeddings.append(
                    Embedding(
                        embedding=emb.embedding,
                        metadata={
                            "prompt_tokens": embedding.usage.prompt_tokens,
                            "total_tokens": embedding.usage.total_tokens,
                        },
                    )
                )
            bar.update(n=len(batch))

        return [
            Embedding(
                embedding=emb.embedding,
                metadata={
                    "prompt_tokens": embedding.usage.prompt_tokens,
                    "total_tokens": embedding.usage.total_tokens,
                },
            )
            for emb in embeddings
        ]
