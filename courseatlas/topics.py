import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from courseatlas import chunking, config, db, parsing
from courseatlas.parsing import PageContent

PAGE_CHAR_LIMIT = 700
BATCH_CHAR_LIMIT = 9000
MAX_TOPIC_NAME = 80

_SYSTEM_PROMPT = """You extract study topics from course material for a student.
You get numbered pages from one document. Return the specific concepts the pages teach.

Rules:
- Use short, specific topic names a student would recognize, like "Carry Look-Ahead Adders" or "Virtual Memory Paging".
- Skip pages that are only administrative: title slides, agendas, "Questions?", grading, office hours, homework due dates. A review page that teaches content still counts.
- Cover every page that teaches something. Do not stop early.
- Keep one topic per concept. Fold sub-points of a concept, like its delay, cost, or an example, into that concept's topic instead of making a new one.
- Consecutive pages about the same concept belong to one topic.
- Only use page numbers that appear in the input.
- A topic can span several pages, and a page can belong to more than one topic.

Respond with JSON only, in this shape:
{"topics": [{"name": "Topic Name", "pages": [1, 2]}]}"""

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


@dataclass
class ExtractedTopic:
    name: str
    pages: list[int] = field(default_factory=list)


def make_json_llm() -> ChatOllama:
    kwargs = {}
    if config.OLLAMA_MODEL.startswith("qwen3"):
        kwargs["reasoning"] = False
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        format="json",
        num_ctx=8192,
        **kwargs,
    )


def _batch_pages(pages: list[PageContent]) -> list[list[PageContent]]:
    batches: list[list[PageContent]] = []
    current: list[PageContent] = []
    size = 0
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        trimmed = PageContent(page=page.page, text=text[:PAGE_CHAR_LIMIT])
        if current and size + len(trimmed.text) > BATCH_CHAR_LIMIT:
            batches.append(current)
            current, size = [], 0
        current.append(trimmed)
        size += len(trimmed.text)
    if current:
        batches.append(current)
    return batches


def _format_batch(batch: list[PageContent]) -> str:
    return "\n\n".join(f"### Page {p.page}\n{p.text}" for p in batch)


def parse_topics(raw: str, valid_pages: set[int]) -> list[ExtractedTopic]:
    """Turn the model's JSON reply into topics, dropping anything malformed or out of range."""
    text = _THINK_RE.sub("", raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    items = data.get("topics", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    topics: list[ExtractedTopic] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        name = " ".join(name.split())
        if not name or len(name) > MAX_TOPIC_NAME:
            continue
        pages = []
        for p in item.get("pages") or []:
            try:
                p = int(p)
            except (TypeError, ValueError):
                continue
            if p in valid_pages and p not in pages:
                pages.append(p)
        if pages:
            topics.append(ExtractedTopic(name=name, pages=sorted(pages)))
    return topics


def _base_name(name: str) -> str:
    """Drop a trailing qualifier, so 'Carry Look-Ahead: Delay' becomes 'carry look-ahead'."""
    return re.split(r"\s*(?::|\s-\s|\s–\s|\()", name, maxsplit=1)[0].strip().lower()


def _merge(topics: list[ExtractedTopic]) -> list[ExtractedTopic]:
    """Merge duplicates and sub-topics like "X: Delay" into the topic "X ..." they belong to."""
    merged: dict[str, ExtractedTopic] = {}
    for t in topics:
        key = t.name.lower()
        if key in merged:
            merged[key].pages = sorted(set(merged[key].pages) | set(t.pages))
        else:
            merged[key] = ExtractedTopic(name=t.name, pages=list(t.pages))

    result = list(merged.values())
    for sub in list(result):
        if _base_name(sub.name) == sub.name.lower():
            continue
        base = _base_name(sub.name)
        parent = next(
            (t for t in result if t is not sub and (t.name.lower() == base or t.name.lower().startswith(base + " "))),
            None,
        )
        if parent:
            parent.pages = sorted(set(parent.pages) | set(sub.pages))
            result.remove(sub)
    return result


def extract_topics(pages: list[PageContent], llm: ChatOllama | None = None) -> list[ExtractedTopic]:
    """Ask the LLM for the topics in a document. Connection errors are raised to the caller."""
    batches = _batch_pages(pages)
    if not batches:
        return []

    llm = llm or make_json_llm()
    found: list[ExtractedTopic] = []
    for batch in batches:
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=_format_batch(batch)),
            ]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        found.extend(parse_topics(raw, {p.page for p in batch}))
    return _merge(found)


def save_topics(
    class_id: str,
    doc_id: str,
    pages: list[PageContent],
    topics: list[ExtractedTopic],
) -> int:
    """Store topics for a document and link them to its chunks. Returns how many topics were linked.

    The document row must already exist in the database.
    """
    chunks = chunking.chunk_pages(class_id=class_id, doc_id=doc_id, document="", pages=pages)
    chunk_ids_by_page: dict[int, list[str]] = {}
    for chunk in chunks:
        chunk_ids_by_page.setdefault(chunk.page, []).append(chunk.chunk_id)

    linked = 0
    for topic in topics:
        pages_with_chunks = [p for p in topic.pages if p in chunk_ids_by_page]
        if not pages_with_chunks:
            continue
        topic_id = db.add_topic(class_id, topic.name)
        for page in pages_with_chunks:
            for chunk_id in chunk_ids_by_page[page]:
                db.link_chunk_topic(topic_id, chunk_id, doc_id, page)
        linked += 1
    return linked


def extract_and_save(class_id: str, doc_id: str, file_path: str | Path) -> int:
    pages = parsing.extract_pages(file_path)
    topics = extract_topics(pages)
    return save_topics(class_id, doc_id, pages, topics)
