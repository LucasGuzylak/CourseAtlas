import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from courseatlas import db, topics
from courseatlas.parsing import PageContent


def _fake_llm(*replies: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content=r) for r in replies]
    return llm


class ParseTopicsTests(unittest.TestCase):
    def test_valid_reply(self) -> None:
        raw = '{"topics": [{"name": "Carry Look-Ahead Adders", "pages": [2, 3]}]}'
        result = topics.parse_topics(raw, {1, 2, 3})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "Carry Look-Ahead Adders")
        self.assertEqual(result[0].pages, [2, 3])

    def test_drops_bad_pages_and_names(self) -> None:
        raw = """<think>hmm</think>
        {"topics": [
            {"name": "Pipelining", "pages": [1, 9, "2", 2, null]},
            {"name": "", "pages": [1]},
            {"name": "Hazards", "pages": [42]},
            {"pages": [1]},
            "junk"
        ]}"""
        result = topics.parse_topics(raw, {1, 2})
        self.assertEqual([(t.name, t.pages) for t in result], [("Pipelining", [1, 2])])

    def test_json_wrapped_in_text(self) -> None:
        raw = 'Here you go: {"topics": [{"name": "TLB", "pages": [1]}]} hope that helps'
        self.assertEqual(topics.parse_topics(raw, {1})[0].name, "TLB")

    def test_garbage_returns_empty(self) -> None:
        self.assertEqual(topics.parse_topics("not json at all", {1}), [])
        self.assertEqual(topics.parse_topics('{"topics": "nope"}', {1}), [])


class ExtractTopicsTests(unittest.TestCase):
    def test_skips_empty_pages_and_batches_long_documents(self) -> None:
        pages = [PageContent(page=i, text="x" * 3000) for i in range(1, 21)]
        pages.append(PageContent(page=21, text="   "))
        batches = topics._batch_pages(pages)
        self.assertGreater(len(batches), 1)
        self.assertTrue(all(len(p.text) <= topics.PAGE_CHAR_LIMIT for b in batches for p in b))
        self.assertNotIn(21, [p.page for b in batches for p in b])

    def test_merges_same_topic_across_batches(self) -> None:
        pages = [PageContent(page=i, text="x" * 3000) for i in range(1, 30)]
        n_batches = len(topics._batch_pages(pages))
        first = '{"topics": [{"name": "Paging", "pages": [1, 2]}]}'
        rest = '{"topics": [{"name": "paging", "pages": [%d]}, {"name": "TLB", "pages": [%d]}]}'
        replies = [first]
        for b in topics._batch_pages(pages)[1:]:
            replies.append(rest % (b[0].page, b[0].page))
        llm = _fake_llm(*replies)

        result = topics.extract_topics(pages, llm=llm)

        self.assertEqual(llm.invoke.call_count, n_batches)
        by_name = {t.name: t.pages for t in result}
        self.assertEqual(set(by_name), {"Paging", "TLB"})
        self.assertEqual(by_name["Paging"][:2], [1, 2])
        self.assertEqual(len(by_name["Paging"]), 2 + n_batches - 1)

    def test_no_text_means_no_llm_call(self) -> None:
        llm = _fake_llm()
        self.assertEqual(topics.extract_topics([PageContent(page=1, text="")], llm=llm), [])
        llm.invoke.assert_not_called()


class TopicStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db = db.DB_PATH
        db.DB_PATH = Path(self.tmp.name) / "test.db"
        db.init_db()
        self.class_id = db.create_class("Computer Architecture")
        self.doc_id = db.new_id()
        db.add_document(self.class_id, self.doc_id, "Lecture_03.pdf", "/tmp/x.pdf", 3)
        self.pages = [
            PageContent(page=1, text="Ripple-carry adders"),
            PageContent(page=2, text="Carry look-ahead adders " * 100),
            PageContent(page=3, text=""),
        ]

    def tearDown(self) -> None:
        db.DB_PATH = self.original_db
        self.tmp.cleanup()

    def test_save_links_every_chunk_on_a_page(self) -> None:
        found = [
            topics.ExtractedTopic("Carry Look-Ahead Adders", [2]),
            topics.ExtractedTopic("Empty Page Topic", [3]),
        ]
        linked = topics.save_topics(self.class_id, self.doc_id, self.pages, found)
        self.assertEqual(linked, 1)

        [topic] = db.list_topics(self.class_id)
        self.assertEqual(topic["name"], "Carry Look-Ahead Adders")
        self.assertEqual(topic["num_pages"], 1)
        self.assertEqual(
            db.get_topic_pages(topic["id"]),
            [{"doc_id": self.doc_id, "document": "Lecture_03.pdf", "page": 2}],
        )
        with sqlite3.connect(db.DB_PATH) as conn:
            n_links = conn.execute("SELECT COUNT(*) FROM chunk_topics").fetchone()[0]
        self.assertGreater(n_links, 1)

    def test_topic_names_are_case_insensitive(self) -> None:
        a = db.add_topic(self.class_id, "Pipelining")
        b = db.add_topic(self.class_id, "  pipelining ")
        self.assertEqual(a, b)
        self.assertEqual(len(db.list_topics(self.class_id)), 1)

    def test_weak_topics_keep_marking_order(self) -> None:
        ids = [db.add_topic(self.class_id, n) for n in ("Alpha", "Beta", "Gamma")]
        db.set_weak(ids[2])
        db.set_weak(ids[0])
        self.assertEqual([t["name"] for t in db.get_weak_topics(self.class_id)], ["Gamma", "Alpha"])

        db.set_weak(ids[2], False)
        db.set_weak(ids[2])
        self.assertEqual([t["name"] for t in db.get_weak_topics(self.class_id)], ["Alpha", "Gamma"])

    def test_deleting_document_cleans_up_topics(self) -> None:
        self.assertEqual(len(db.documents_without_topics(self.class_id)), 1)
        topics.save_topics(self.class_id, self.doc_id, self.pages, [topics.ExtractedTopic("Adders", [1])])
        self.assertEqual(db.documents_without_topics(self.class_id), [])

        db.delete_document(self.doc_id)
        db.delete_orphan_topics(self.class_id)
        self.assertEqual(db.list_topics(self.class_id), [])

    def test_init_db_adds_weak_order_to_old_databases(self) -> None:
        old_path = Path(self.tmp.name) / "old.db"
        with sqlite3.connect(old_path) as conn:
            conn.execute(
                "CREATE TABLE topics (id TEXT PRIMARY KEY, class_id TEXT NOT NULL, name TEXT NOT NULL, "
                "is_weak INTEGER NOT NULL DEFAULT 0, UNIQUE (class_id, name))"
            )
        db.DB_PATH = old_path
        db.init_db()
        with sqlite3.connect(old_path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(topics)")}
        self.assertIn("weak_order", cols)


if __name__ == "__main__":
    unittest.main()
