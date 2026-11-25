from dataclasses import dataclass, field
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass
class Chunk:
    chunk_id: str  # doc_id:page:n
    class_id: str
    doc_id: str
    document: str
    page: int  # 1-based
    text: str
    score: float | None = None


@dataclass
class Citation:
    chunk_id: str
    document: str
    page: int


@dataclass
class AnswerResult:
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class IngestResult:
    doc_id: str
    num_pages: int
    num_chunks: int
