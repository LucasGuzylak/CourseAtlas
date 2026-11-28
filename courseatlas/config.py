import os
from pathlib import Path

from courseatlas import db


def get_chroma_dir() -> Path:
    """Return the persistent ChromaDB directory, respecting env overrides."""
    override = os.environ.get("COURSEATLAS_CHROMA_DIR")
    if override:
        return Path(override)
    return db.DB_PATH.parent / "chroma"


CHROMA_COLLECTION = os.environ.get("COURSEATLAS_CHROMA_COLLECTION", "courseatlas_chunks")

# Embeddings & optional reranker
EMBEDDING_MODEL = os.environ.get("COURSEATLAS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.environ.get("COURSEATLAS_RERANKER_MODEL", "BAAI/bge-reranker-base")
USE_RERANKER = os.environ.get("COURSEATLAS_USE_RERANKER", "0").lower() in {"1", "true", "yes"}

# Chunking (tuned so a normal slide/page stays a single chunk; long pages split cleanly)
CHUNK_SIZE = int(os.environ.get("COURSEATLAS_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("COURSEATLAS_CHUNK_OVERLAP", "150"))

# Retrieval
DEFAULT_TOP_K = int(os.environ.get("COURSEATLAS_TOP_K", "5"))
RERANK_CANDIDATES = int(os.environ.get("COURSEATLAS_RERANK_CANDIDATES", "20"))

# Local LLM via Ollama
OLLAMA_MODEL = os.environ.get("COURSEATLAS_LLM_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.environ.get("COURSEATLAS_LLM_TEMPERATURE", "0.2"))
MAX_HISTORY_TURNS = int(os.environ.get("COURSEATLAS_MAX_HISTORY_TURNS", "6"))
