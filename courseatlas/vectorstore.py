from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from courseatlas import config
from courseatlas.schemas import Chunk

_clients: dict[str, chromadb.ClientAPI] = {}


def _get_collection(chroma_dir: Path | None = None, collection_name: str | None = None) -> Collection:
    directory = (chroma_dir or config.get_chroma_dir()).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    key = str(directory)
    if key not in _clients:
        _clients[key] = chromadb.PersistentClient(path=key)
    name = collection_name or config.CHROMA_COLLECTION
    return _clients[key].get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_index(chunk_id: str) -> int:
    try:
        return int(chunk_id.rsplit(":", 1)[-1])
    except ValueError:
        return 0


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Persist chunks and their embeddings in ChromaDB with class/document/page metadata."""
    if not chunks:
        return
    collection = _get_collection()
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "class_id": c.class_id,
                "doc_id": c.doc_id,
                "document": c.document,
                "page": int(c.page),
                "chunk_index": _chunk_index(c.chunk_id),
            }
            for c in chunks
        ],
    )


def delete_document(class_id: str, doc_id: str) -> None:
    """Remove all chunks belonging to a single document within a class."""
    collection = _get_collection()
    if collection.count() == 0:
        return
    collection.delete(where={"$and": [{"class_id": class_id}, {"doc_id": doc_id}]})


def query_chunks(class_id: str, query_embedding: list[float], k: int = 5) -> list[Chunk]:
    """Retrieve the top-k most similar chunks restricted strictly to class_id."""
    collection = _get_collection()
    total = collection.count()
    if total == 0 or k <= 0:
        return []

    n_results = min(k, total)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"class_id": class_id},
        include=["documents", "metadatas", "distances"],
    )

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    chunks: list[Chunk] = []
    for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
        similarity = round(1.0 - float(dist), 4) if dist is not None else None
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                class_id=str(meta["class_id"]),
                doc_id=str(meta["doc_id"]),
                document=str(meta["document"]),
                page=int(meta["page"]),
                text=text or "",
                score=similarity,
            )
        )
    return chunks


def get_document_chunks(class_id: str, doc_id: str, page: int | None = None) -> list[Chunk]:
    """Fetch stored chunks for a document (optionally filtered to one 1-based page)."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    filters: list[dict] = [{"class_id": class_id}, {"doc_id": doc_id}]
    if page is not None:
        filters.append({"page": int(page)})

    raw = collection.get(
        where={"$and": filters},
        include=["documents", "metadatas"],
    )
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    chunks: list[Chunk] = []
    for chunk_id, text, meta in zip(ids, docs, metas):
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                class_id=str(meta["class_id"]),
                doc_id=str(meta["doc_id"]),
                document=str(meta["document"]),
                page=int(meta["page"]),
                text=text or "",
            )
        )
    chunks.sort(key=lambda c: (c.page, _chunk_index(c.chunk_id)))
    return chunks
