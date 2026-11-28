import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from courseatlas import config, db, evaluation
from courseatlas.schemas import AnswerResult, Chunk, Citation


def _chunk(doc_id: str, page: int) -> Chunk:
    return Chunk(f"{doc_id}:{page}:0", "cls", doc_id, f"{doc_id}.pdf", page, f"text {page}")


QUESTION = {"question": "q?", "doc_id": "d1", "pages": [4, 5], "document": "d1.pdf", "reference_answer": "ref"}


class RetrievalMetricTests(unittest.TestCase):
    def test_first_relevant_rank(self) -> None:
        chunks = [_chunk("d1", 2), _chunk("d2", 4), _chunk("d1", 5)]
        self.assertEqual(evaluation.first_relevant_rank(chunks, QUESTION), 3)
        self.assertIsNone(evaluation.first_relevant_rank(chunks[:2], QUESTION))

    def test_score_retrieval_summary(self) -> None:
        results = {
            "hit first": [_chunk("d1", 4)],
            "hit third": [_chunk("d1", 1), _chunk("d1", 2), _chunk("d1", 5)],
            "miss": [_chunk("d2", 4)],
        }
        questions = [{**QUESTION, "question": q} for q in results]
        with patch.object(evaluation.pipeline, "retrieve", side_effect=lambda c, q, k: results[q]):
            res = evaluation.score_retrieval("cls", questions, k=5)
        s = res["summary"]
        self.assertAlmostEqual(s["hit@1"], 1 / 3)
        self.assertAlmostEqual(s["hit@3"], 2 / 3)
        self.assertAlmostEqual(s["hit@5"], 2 / 3)
        self.assertAlmostEqual(s["mrr"], (1 + 1 / 3) / 3)
        self.assertEqual([r["rank"] for r in res["rows"]], [1, 3, None])

    def test_reranker_switch_restores_setting(self) -> None:
        before = config.USE_RERANKER
        with evaluation.reranker(not before):
            self.assertEqual(config.USE_RERANKER, not before)
        self.assertEqual(config.USE_RERANKER, before)


class AnswerScoringTests(unittest.TestCase):
    def test_scores_citations_and_judges(self) -> None:
        chunks = [_chunk("d1", 4), _chunk("d1", 9)]
        answer = AnswerResult(
            text="Claim A [1]. Claim B [2].",
            citations=[Citation("d1:4:0", "d1.pdf", 4), Citation("d1:9:0", "d1.pdf", 9)],
        )
        judge = MagicMock()
        judge.invoke.side_effect = [
            MagicMock(content='{"claims": [{"claim": "A", "supported": true}, {"claim": "B", "supported": false}]}'),
            MagicMock(content='{"score": 0.5, "reason": "misses a detail"}'),
        ]
        with (
            patch.object(evaluation, "check_ollama"),
            patch.object(evaluation.pipeline, "retrieve", return_value=chunks),
            patch.object(evaluation.llm, "generate_answer", return_value=answer),
            patch.object(evaluation.topics, "make_json_llm", return_value=judge),
        ):
            res = evaluation.score_answers("cls", [QUESTION])

        row = res["rows"][0]
        self.assertEqual(row["citation_precision"], 0.5)
        self.assertTrue(row["cited_correct_page"])
        self.assertEqual(row["faithfulness"], 0.5)
        self.assertEqual(row["unsupported_claims"], "B")
        self.assertEqual(row["correctness"], 0.5)
        self.assertEqual(res["summary"]["hallucination_rate"], 1.0)
        self.assertEqual(res["summary"]["answer_accuracy"], 0.5)

    def test_without_judge(self) -> None:
        answer = AnswerResult(text="No markers here.", citations=[Citation("d2:1:0", "d2.pdf", 1)])
        with (
            patch.object(evaluation, "check_ollama"),
            patch.object(evaluation.pipeline, "retrieve", return_value=[]),
            patch.object(evaluation.llm, "generate_answer", return_value=answer),
        ):
            res = evaluation.score_answers("cls", [QUESTION], use_judge=False)
        self.assertEqual(res["rows"][0]["citation_precision"], 0.0)
        self.assertFalse(res["rows"][0]["has_inline_citations"])
        self.assertIsNone(res["summary"]["faithfulness"])

    def test_ollama_down_raises(self) -> None:
        with patch.object(evaluation.urllib.request, "urlopen", side_effect=OSError("refused")):
            with self.assertRaises(RuntimeError):
                evaluation.check_ollama()


class TopicScoringTests(unittest.TestCase):
    def test_matching(self) -> None:
        self.assertTrue(evaluation.topics_match("Carry Look-Ahead Adders", "carry look-ahead adder"))
        self.assertTrue(evaluation.topics_match("Pipelining", "Intro to Pipelining"))
        self.assertTrue(evaluation.topics_match("Data Hazards", "Data Hazards and Stalls"))
        self.assertFalse(evaluation.topics_match("Data Hazards", "Control Hazards"))
        self.assertFalse(evaluation.topics_match("Caches", "Virtual Memory"))

    def test_score_topic_list(self) -> None:
        res = evaluation.score_topic_list(
            ["Ripple-Carry Adders", "Carry Look-Ahead Adders", "Branch Prediction"],
            ["Ripple Carry Adders", "Carry Look-Ahead Adders", "Half Adder", "Forwarding"],
        )
        self.assertEqual(res["precision"], 0.5)
        self.assertAlmostEqual(res["recall"], 2 / 3)
        self.assertEqual(res["missed"], ["Branch Prediction"])
        self.assertEqual(res["extra"], ["Half Adder", "Forwarding"])


class EvalStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db = db.DB_PATH
        db.DB_PATH = Path(self.tmp.name) / "test.db"
        db.init_db()
        self.class_id = db.create_class("Arch")
        self.doc_id = db.new_id()
        db.add_document(self.class_id, self.doc_id, "Lecture.pdf", "/tmp/x.pdf", 10)

    def tearDown(self) -> None:
        db.DB_PATH = self.original_db
        self.tmp.cleanup()

    def test_questions_round_trip_and_cascade(self) -> None:
        db.add_eval_question(self.class_id, " What is CLA? ", self.doc_id, [5, 4, 4], "ref")
        [q] = db.list_eval_questions(self.class_id)
        self.assertEqual(q["question"], "What is CLA?")
        self.assertEqual(q["pages"], [4, 5])
        self.assertEqual(q["document"], "Lecture.pdf")

        db.delete_document(self.doc_id)
        self.assertEqual(db.list_eval_questions(self.class_id), [])

    def test_topic_labels_replace(self) -> None:
        db.set_topic_labels(self.doc_id, ["Adders", "  ", "Pipelining"])
        db.set_topic_labels(self.doc_id, ["Caches", "Pipelining", ""])
        self.assertEqual(db.get_topic_labels(self.doc_id), ["Caches", "Pipelining"])

    def test_score_topics_uses_labels_and_extracted(self) -> None:
        topic_id = db.add_topic(self.class_id, "Pipelining")
        db.link_chunk_topic(topic_id, f"{self.doc_id}:2:0", self.doc_id, 2)
        db.set_topic_labels(self.doc_id, ["Pipelining", "Caches"])
        res = evaluation.score_topics(self.class_id)
        self.assertEqual(res["summary"]["precision"], 1.0)
        self.assertEqual(res["summary"]["recall"], 0.5)

    def test_save_results(self) -> None:
        out = Path(self.tmp.name) / "eval"
        path = evaluation.save_results("retrieval", {"summary": {"mrr": 0.5}, "rows": [{"question": "q", "rank": 1}]}, out)
        self.assertEqual(json.loads(path.read_text())["summary"]["mrr"], 0.5)
        self.assertTrue(path.with_suffix(".csv").exists())


if __name__ == "__main__":
    unittest.main()
