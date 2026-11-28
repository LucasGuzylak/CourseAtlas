from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from courseatlas import config
from courseatlas.schemas import Chunk

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _load_reranker_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def embed_documents(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Embed document chunks into normalized dense vectors."""
    if not texts:
        return []
    name = model_name or config.EMBEDDING_MODEL
    model = _load_embedding_model(name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str, model_name: str | None = None) -> list[float]:
    """Embed a user query, adding the recommended BGE instruction prefix when applicable."""
    name = model_name or config.EMBEDDING_MODEL
    model = _load_embedding_model(name)
    text = f"{_BGE_QUERY_PREFIX}{query}" if "bge-" in name.lower() else query
    vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()


def rerank(query: str, chunks: list[Chunk], top_k: int, model_name: str | None = None) -> list[Chunk]:
    """Rerank retrieved chunks with a cross-encoder and return the top_k results."""
    if not chunks:
        return []
    name = model_name or config.RERANKER_MODEL
    model = _load_reranker_model(name)
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)
    result: list[Chunk] = []
    for chunk, score in ranked[:top_k]:
        result.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                class_id=chunk.class_id,
                doc_id=chunk.doc_id,
                document=chunk.document,
                page=chunk.page,
                text=chunk.text,
                score=round(float(score), 4),
            )
        )
    return result
