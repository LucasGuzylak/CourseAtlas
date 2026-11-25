# TODO: fake data for now, swap in the real pipeline but keep the signatures

from pathlib import Path

from courseatlas.schemas import AnswerResult, Chunk, Citation, IngestResult, Message

_FAKE_PAGES_PER_DOC = 10

_store: dict[str, dict[str, str]] = {}


def ingest(class_id: str, doc_id: str, file_path: str | Path) -> IngestResult:
    _store.setdefault(class_id, {})[doc_id] = Path(file_path).name
    return IngestResult(doc_id=doc_id, num_pages=_FAKE_PAGES_PER_DOC, num_chunks=_FAKE_PAGES_PER_DOC)


def delete_document(class_id: str, doc_id: str) -> None:
    _store.get(class_id, {}).pop(doc_id, None)


def retrieve(class_id: str, query: str, k: int = 5) -> list[Chunk]:
    docs = _store.get(class_id) or {"demo-doc": "Lecture_03.pdf"}
    chunks: list[Chunk] = []
    for doc_id, filename in docs.items():
        for page in range(1, _FAKE_PAGES_PER_DOC + 1):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{page}:0",
                    class_id=class_id,
                    doc_id=doc_id,
                    document=filename,
                    page=page,
                    text=f"[stub] Content of {filename}, page {page}, related to: {query}",
                    score=round(1.0 - 0.05 * page, 2),
                )
            )
    chunks.sort(key=lambda c: c.score or 0.0, reverse=True)
    return chunks[:k]


def answer(class_id: str, query: str, history: list[Message]) -> AnswerResult:
    chunks = retrieve(class_id, query, k=2)
    citations = [Citation(chunk_id=c.chunk_id, document=c.document, page=c.page) for c in chunks]
    text = (
        f"[stub] Here's an explanation of \"{query}\" [1]. "
        f"More detail is on the next slide [2]. "
        f"({len(history)} earlier messages in this chat.)"
    )
    return AnswerResult(text=text, citations=citations)
