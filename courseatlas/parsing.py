from dataclasses import dataclass
from pathlib import Path

import pymupdf
from pptx import Presentation

SUPPORTED_EXTENSIONS = {".pdf", ".pptx"}


@dataclass
class PageContent:
    page: int  # 1-based page or slide number
    text: str


def _clean_text(text: str) -> str:
    """Normalize line endings and trim trailing whitespace per line without altering content."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    # Collapse runs of 3+ blank lines down to a single blank line
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _extract_pdf(file_path: Path) -> list[PageContent]:
    pages: list[PageContent] = []
    with pymupdf.open(file_path) as doc:
        for idx, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            pages.append(PageContent(page=idx, text=_clean_text(raw_text)))
    return pages


def _extract_pptx(file_path: Path) -> list[PageContent]:
    prs = Presentation(str(file_path))
    pages: list[PageContent] = []
    for idx, slide in enumerate(prs.slides, start=1):
        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                paragraphs = [p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()]
                if paragraphs:
                    parts.append("\n".join(paragraphs))
            elif shape.has_table:
                rows: list[str] = []
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    parts.append("\n".join(rows))
        pages.append(PageContent(page=idx, text=_clean_text("\n".join(parts))))
    return pages


def extract_pages(file_path: str | Path) -> list[PageContent]:
    """Extract plain text per page (PDF) or per slide (PPTX) with 1-based numbering."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".pptx":
        return _extract_pptx(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}")


def extract_single_page(file_path: str | Path, page: int) -> PageContent:
    """Extract text for a single 1-based page/slide."""
    path = Path(file_path)
    if page < 1:
        raise ValueError("Page number must be >= 1.")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(path) as doc:
            if page > len(doc):
                raise ValueError(f"Page {page} out of range (1..{len(doc)}).")
            raw_text = doc[page - 1].get_text("text")
            return PageContent(page=page, text=_clean_text(raw_text))
    if suffix == ".pptx":
        pages = _extract_pptx(path)
        if page > len(pages):
            raise ValueError(f"Slide {page} out of range (1..{len(pages)}).")
        return pages[page - 1]
    raise ValueError(f"Unsupported file type '{suffix}'.")


def render_page_image(file_path: str | Path, page: int, dpi: int = 150) -> bytes | None:
    """Render a 1-based PDF page as PNG bytes. Returns None for non-PDF files."""
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return None
    if page < 1:
        raise ValueError("Page number must be >= 1.")

    with pymupdf.open(path) as doc:
        if page > len(doc):
            raise ValueError(f"Page {page} out of range (1..{len(doc)}).")
        pixmap = doc[page - 1].get_pixmap(dpi=dpi)
        return pixmap.tobytes("png")
