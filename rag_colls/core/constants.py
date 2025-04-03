import os

DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ["true", "1"]

OPENAI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

HF_EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
]

DEFAULT_HF_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

DEFAULT_OPENAI_MODEL = "openai/gpt-4o-mini"
