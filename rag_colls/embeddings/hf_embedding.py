from tqdm import tqdm
from typing import Optional


from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv

from rag_colls.types.embedding import Embedding
from rag_colls.types.core.document import Document
from rag_colls.core.base.embeddings.base import BaseEmbedding
from rag_colls.core.utils import check_torch_device
from rag_colls.core.constants import (
    HF_EMBEDDING_MODELS,
    DEFAULT_HF_EMBEDDING_MODEL,
)

load_dotenv()


class HuggingFaceEmbedding(BaseEmbedding):
    model = None
    tokenizer = None

    def __init__(
        self,
        model_name: str | None = None,
        pooling: str = "cls",
        max_length: Optional[int] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        normalize: bool = True,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        device: Optional[str] = "auto",
        **model_kwargs,
    ):
        """
        Initialize the HuggingFace embedding model.

        Args:
            model_name (str): The name of the HuggingFace model to use. Can only be chosen from
                `[sentence-transformers/all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2, sentence-transformers/all-distilroberta-v1]`.
        """
        if not model_name:
            model_name = DEFAULT_HF_EMBEDDING_MODEL

        assert model_name in HF_EMBEDDING_MODELS, (
            f"Model {model_name} is not supported."
        )

        self.pooling = pooling
        self.max_length = max_length
        self.normalize = normalize
        self.cache_folder = cache_folder
        self.trust_remote_code = trust_remote_code
        self.device = check_torch_device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_folder,
            device_map=self.device,
            **model_kwargs,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_folder,
            device_map=self.device,
            **model_kwargs,
        )
        self.model.eval()  # Set model to evaluation mode

    def __str__(self):
        return f"HFEmbedding(model_name={self.model_name})"

    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling of token embeddings to get sentence embeddings.
        """
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _cls_pooling(self, model_output, attention_mask):
        """
        CLS pooling of token embeddings to get sentence embeddings.
        """
        return model_output.pooler_output

    def _get_query_embedding(self, query: str, **kwargs) -> Embedding:
        """
        Returns the embedding of the query.

        Args:
            query (str): The query to be embedded.

        Returns:
            Embedding: The embedding object of the query.
        """
        # Tokenize the query
        encoded_input = self.tokenizer(
            query, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        if self.pooling == "cls":
            embedding = self._cls_pooling(model_output, encoded_input["attention_mask"])
        else:
            embedding = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

        # Get token counts
        tokens = self.tokenizer.encode(query, add_special_tokens=False)
        total_tokens = self.tokenizer.encode(query, add_special_tokens=True)

        return Embedding(
            embedding=embedding[0].tolist(),
            metadata={
                "prompt_tokens": len(tokens),
                "total_tokens": len(total_tokens),
            },
        )

    def _get_document_embedding(self, document: Document, **kwargs) -> Embedding:
        """
        Returns the embedding of the document.
        Args:
            document (Document): The document to be embedded.

        Returns:
            Embedding: The embedding object of the document.
        """
        # Tokenize the document
        encoded_input = self.tokenizer(
            document.document, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        if self.pooling == "cls":
            embedding = self._cls_pooling(model_output, encoded_input["attention_mask"])
        else:
            embedding = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

        # Get token counts
        tokens = self.tokenizer.encode(document.document, add_special_tokens=False)
        total_tokens = self.tokenizer.encode(document.document, add_special_tokens=True)

        return Embedding(
            embedding=embedding[0].tolist(),
            metadata={
                "prompt_tokens": len(tokens),
                "total_tokens": len(total_tokens),
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

            # Tokenize the batch
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            if self.pooling == "cls":
                batch_embeddings = self._cls_pooling(
                    model_output, encoded_input["attention_mask"]
                )
            else:
                batch_embeddings = self._mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )

            # Process each query in the batch
            for query, emb in zip(batch, batch_embeddings):
                tokens = self.tokenizer.encode(query, add_special_tokens=False)
                total_tokens = self.tokenizer.encode(query, add_special_tokens=True)
                embeddings.append(
                    Embedding(
                        embedding=emb.tolist(),
                        metadata={
                            "prompt_tokens": len(tokens),
                            "total_tokens": len(total_tokens),
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

            # Tokenize the batch
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            if self.pooling == "cls":
                batch_embeddings = self._cls_pooling(
                    model_output, encoded_input["attention_mask"]
                )
            else:
                batch_embeddings = self._mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )

            # Process each document in the batch
            for doc, emb in zip(batch, batch_embeddings):
                tokens = self.tokenizer.encode(doc, add_special_tokens=False)
                total_tokens = self.tokenizer.encode(doc, add_special_tokens=True)
                embeddings.append(
                    Embedding(
                        embedding=emb.tolist(),
                        metadata={
                            "prompt_tokens": len(tokens),
                            "total_tokens": len(total_tokens),
                        },
                    )
                )
            bar.update(n=len(batch))

        return embeddings
