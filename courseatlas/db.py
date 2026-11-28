import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from courseatlas.schemas import Citation

DB_PATH = Path(os.environ.get("COURSEATLAS_DB", Path(__file__).resolve().parent.parent / "data" / "courseatlas.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS classes (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    class_id    TEXT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    path        TEXT NOT NULL,
    num_pages   INTEGER,
    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id   TEXT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    citations  TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS topics (
    id       TEXT PRIMARY KEY,
    class_id TEXT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    name     TEXT NOT NULL,
    is_weak  INTEGER NOT NULL DEFAULT 0,
    weak_order INTEGER,
    UNIQUE (class_id, name)
);

CREATE TABLE IF NOT EXISTS chunk_topics (
    chunk_id TEXT NOT NULL,
    topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    doc_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page     INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_class ON chat_messages(class_id, id);
CREATE INDEX IF NOT EXISTS idx_documents_class ON documents(class_id);
CREATE INDEX IF NOT EXISTS idx_chunk_topics_topic ON chunk_topics(topic_id);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def new_id() -> str:
    return uuid.uuid4().hex


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(SCHEMA)
        topic_cols = {r["name"] for r in conn.execute("PRAGMA table_info(topics)")}
        if "weak_order" not in topic_cols:
            conn.execute("ALTER TABLE topics ADD COLUMN weak_order INTEGER")


# Classes

def create_class(name: str) -> str:
    class_id = new_id()
    with _connect() as conn:
        conn.execute("INSERT INTO classes (id, name) VALUES (?, ?)", (class_id, name.strip()))
    return class_id


def list_classes() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM classes ORDER BY created_at, name").fetchall()
    return [dict(r) for r in rows]


def rename_class(class_id: str, name: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE classes SET name = ? WHERE id = ?", (name.strip(), class_id))


def delete_class(class_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM classes WHERE id = ?", (class_id,))


# Documents

def add_document(class_id: str, doc_id: str, filename: str, path: str | Path, num_pages: int | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, class_id, filename, path, num_pages) VALUES (?, ?, ?, ?, ?)",
            (doc_id, class_id, filename, str(path), num_pages),
        )


def get_document(doc_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def list_documents(class_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE class_id = ? ORDER BY uploaded_at, filename", (class_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# Chat history

def add_message(class_id: str, role: str, content: str, citations: list[Citation] | None = None) -> None:
    cites = json.dumps([asdict(c) for c in citations or []])
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_messages (class_id, role, content, citations) VALUES (?, ?, ?, ?)",
            (class_id, role, content, cites),
        )


def get_history(class_id: str, limit: int | None = None) -> list[dict]:
    query = "SELECT role, content, citations, created_at FROM chat_messages WHERE class_id = ? ORDER BY id DESC"
    params: tuple = (class_id,)
    if limit is not None:
        query += " LIMIT ?"
        params += (limit,)
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    history = []
    for r in reversed(rows):
        msg = dict(r)
        msg["citations"] = [Citation(**c) for c in json.loads(msg["citations"])]
        history.append(msg)
    return history


def clear_history(class_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM chat_messages WHERE class_id = ?", (class_id,))


# Topics

def add_topic(class_id: str, name: str) -> str:
    """Return the id of the class's topic with this name (case-insensitive), creating it if needed."""
    name = " ".join(name.split())
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM topics WHERE class_id = ? AND lower(name) = lower(?)", (class_id, name)
        ).fetchone()
        if row:
            return row["id"]
        topic_id = new_id()
        conn.execute("INSERT INTO topics (id, class_id, name) VALUES (?, ?, ?)", (topic_id, class_id, name))
    return topic_id


def list_topics(class_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT t.*, COUNT(DISTINCT ct.doc_id || ':' || ct.page) AS num_pages
            FROM topics t
            LEFT JOIN chunk_topics ct ON ct.topic_id = t.id
            WHERE t.class_id = ?
            GROUP BY t.id
            ORDER BY t.name COLLATE NOCASE
            """,
            (class_id,),
        ).fetchall()
    return [{**dict(r), "is_weak": bool(r["is_weak"])} for r in rows]


def set_weak(topic_id: str, weak: bool = True) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE topics
            SET is_weak = ?,
                weak_order = CASE WHEN ? THEN (SELECT COALESCE(MAX(weak_order), 0) + 1 FROM topics) END
            WHERE id = ?
            """,
            (int(weak), int(weak), topic_id),
        )


def get_weak_topics(class_id: str) -> list[dict]:
    """Weak topics in the order the student marked them."""
    weak = [t for t in list_topics(class_id) if t["is_weak"]]
    return sorted(weak, key=lambda t: t["weak_order"] or 0)


def delete_orphan_topics(class_id: str) -> None:
    """Remove topics that no longer point at any page, e.g. after their document was deleted."""
    with _connect() as conn:
        conn.execute(
            """
            DELETE FROM topics
            WHERE class_id = ? AND id NOT IN (SELECT DISTINCT topic_id FROM chunk_topics)
            """,
            (class_id,),
        )


def documents_without_topics(class_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM documents
            WHERE class_id = ? AND id NOT IN (SELECT DISTINCT doc_id FROM chunk_topics)
            ORDER BY uploaded_at, filename
            """,
            (class_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def link_chunk_topic(topic_id: str, chunk_id: str, doc_id: str, page: int) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO chunk_topics (chunk_id, topic_id, doc_id, page) VALUES (?, ?, ?, ?)",
            (chunk_id, topic_id, doc_id, page),
        )


def get_topic_pages(topic_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ct.doc_id, d.filename AS document, ct.page
            FROM chunk_topics ct
            JOIN documents d ON d.id = ct.doc_id
            WHERE ct.topic_id = ?
            ORDER BY d.filename, ct.page
            """,
            (topic_id,),
        ).fetchall()
    return [dict(r) for r in rows]
