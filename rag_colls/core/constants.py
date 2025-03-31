import os

DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ["true", "1"]

OPENAI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

DEFAULT_OPENAI_MODEL = "openai/gpt-4o-mini"
