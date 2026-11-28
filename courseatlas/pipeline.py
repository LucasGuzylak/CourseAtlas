from dataclasses import dataclass
from pathlib import Path

from courseatlas import chunking, config, db, embeddings, llm, parsing, vectorstore
from courseatlas.schemas import AnswerResult, Chunk, IngestResult, Message


@dataclass
class PageInspection:
    page: int
    document: str
    image_bytes: bytes | None
    extracted_text: str
    chunks: list[Chunk]


def _resolve_document_name(doc_id: str, fallback: str) -> str:
    doc = db.get_document(doc_id)
    if doc and doc.get("filename"):
        return str(doc["filename"])
    return fallback


def ingest(
    class_id: str,
    doc_id: str,
    file_path: str | Path,
    document_name: str | None = None,
) -> IngestResult:
    """Extract text per page/slide, chunk with page metadata, embed, and store in ChromaDB."""
    path = Path(file_path)
    display_name = document_name or _resolve_document_name(doc_id, path.name)

    pages = parsing.extract_pages(path)
    chunks = chunking.chunk_pages(
        class_id=class_id,
        doc_id=doc_id,
        document=display_name,
        pages=pages,
    )

    if chunks:
        vectors = embeddings.embed_documents([c.text for c in chunks])
        vectorstore.upsert_chunks(chunks, vectors)

    return IngestResult(
        doc_id=doc_id,
        num_pages=len(pages),
        num_chunks=len(chunks),
    )


def delete_document(class_id: str, doc_id: str) -> None:
    """Remove all vector chunks for a document from ChromaDB."""
    vectorstore.delete_document(class_id=class_id, doc_id=doc_id)


def retrieve(class_id: str, query: str, k: int = 5) -> list[Chunk]:
    """Retrieve the top-k matching chunks from the given class only."""
    if not query.strip():
        return []

    query_vec = embeddings.embed_query(query)
    fetch_k = config.RERANK_CANDIDATES if config.USE_RERANKER else k
    chunks = vectorstore.query_chunks(class_id=class_id, query_embedding=query_vec, k=fetch_k)

    # Ensure citations display the original uploaded filename even if ingest() was called
    # before db.add_document().
    for chunk in chunks:
        chunk.document = _resolve_document_name(chunk.doc_id, chunk.document)

    if config.USE_RERANKER and chunks:
        chunks = embeddings.rerank(query=query, chunks=chunks, top_k=k)

    return chunks[:k]


def answer(class_id: str, query: str, history: list[Message]) -> AnswerResult:
    """Retrieve grounded course passages for a class and generate a cited answer."""
    chunks = retrieve(class_id=class_id, query=query, k=config.DEFAULT_TOP_K)
    return llm.generate_answer(query=query, chunks=chunks, history=history)


def inspect_page(
    class_id: str,
    doc_id: str,
    file_path: str | Path,
    page: int,
    document_name: str | None = None,
) -> PageInspection:
    """Return the 3-step pipeline view (slide image, extracted text, stored chunks) for one page."""
    path = Path(file_path)
    display_name = document_name or _resolve_document_name(doc_id, path.name)
    page_content = parsing.extract_single_page(path, page)
    image_bytes = parsing.render_page_image(path, page)
    stored_chunks = vectorstore.get_document_chunks(class_id=class_id, doc_id=doc_id, page=page)
    for chunk in stored_chunks:
        chunk.document = display_name

    return PageInspection(
        page=page,
        document=display_name,
        image_bytes=image_bytes,
        extracted_text=page_content.text,
        chunks=stored_chunks,
    )
