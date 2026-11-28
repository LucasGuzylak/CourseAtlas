from langchain_text_splitters import RecursiveCharacterTextSplitter

from courseatlas import config
from courseatlas.parsing import PageContent
from courseatlas.schemas import Chunk


def chunk_pages(
    class_id: str,
    doc_id: str,
    document: str,
    pages: list[PageContent],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Create one chunk per slide/page, splitting only pages that exceed chunk_size.

    Every chunk maps to exactly one 1-based page/slide, with chunk_id formatted as
    ``{doc_id}:{page}:{n}``.
    """
    max_chars = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []
    for page_content in pages:
        text = page_content.text.strip()
        if not text:
            continue

        if len(text) <= max_chars:
            pieces = [text]
        else:
            pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]

        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{page_content.page}:{idx}",
                    class_id=class_id,
                    doc_id=doc_id,
                    document=document,
                    page=page_content.page,
                    text=piece,
                )
            )

    return chunks
