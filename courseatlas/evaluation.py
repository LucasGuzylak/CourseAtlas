import argparse
import csv
import json
import re
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

from langchain_core.messages import HumanMessage, SystemMessage

from courseatlas import config, db, llm, pipeline, topics
from courseatlas.schemas import Chunk

K_VALUES = (1, 3, 5)
RESULTS_DIR = db.DB_PATH.parent / "eval"

_FAITHFULNESS_PROMPT = """You check whether an answer is supported by course excerpts.
Split the answer into its factual claims. Skip greetings, offers to help, and statements that the excerpts don't cover something.
For each claim, decide whether the excerpts support it. General knowledge that the excerpts do not state counts as unsupported.

Respond with JSON only:
{"claims": [{"claim": "short restatement", "supported": true}]}"""

_CORRECTNESS_PROMPT = """You grade an answer to a student's question against a reference answer written by the course staff.
Score 1 if the answer contains the key facts of the reference and does not contradict it.
Score 0.5 if it is partly right or misses important facts.
Score 0 if it is wrong, contradicts the reference, or does not answer.

Respond with JSON only:
{"score": 1, "reason": "one sentence"}"""

_STOPWORDS = {"a", "an", "and", "the", "of", "in", "on", "to", "for", "with", "vs", "intro", "introduction", "basics"}


# Retrieval

@contextmanager
def reranker(enabled: bool):
    previous = config.USE_RERANKER
    config.USE_RERANKER = enabled
    try:
        yield
    finally:
        config.USE_RERANKER = previous


def _relevant(doc_id: str, page: int, question: dict) -> bool:
    return doc_id == question["doc_id"] and page in question["pages"]


def first_relevant_rank(chunks: list[Chunk], question: dict) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if _relevant(chunk.doc_id, chunk.page, question):
            return rank
    return None


def _expected(question: dict) -> str:
    return f"{question['document']} p. {', '.join(map(str, question['pages']))}"


def score_retrieval(class_id: str, questions: list[dict], k: int = 5, use_reranker: bool = False) -> dict:
    rows = []
    with reranker(use_reranker):
        for q in questions:
            chunks = pipeline.retrieve(class_id, q["question"], k=k)
            rank = first_relevant_rank(chunks, q)
            rows.append(
                {
                    "question": q["question"],
                    "expected": _expected(q),
                    "retrieved": "; ".join(f"{c.document} p. {c.page}" for c in chunks),
                    "rank": rank,
                }
            )

    summary = {"questions": len(rows)}
    if rows:
        for n in K_VALUES:
            if n <= k:
                summary[f"hit@{n}"] = float(mean(r["rank"] is not None and r["rank"] <= n for r in rows))
        summary["mrr"] = float(mean(1 / r["rank"] if r["rank"] else 0 for r in rows))
    return {"summary": summary, "rows": rows}


# Answers

def check_ollama() -> None:
    try:
        urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
    except OSError as e:
        raise RuntimeError(f"Ollama isn't reachable at {config.OLLAMA_BASE_URL}. Start it with `ollama serve`.") from e


def _ask_json(judge, system: str, user: str) -> dict:
    response = judge.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = response.content if isinstance(response.content, str) else str(response.content)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    return data if isinstance(data, dict) else {}


def judge_faithfulness(judge, answer: str, chunks: list[Chunk]) -> tuple[float | None, list[str]]:
    """Fraction of the answer's claims the excerpts support, plus the unsupported claims."""
    excerpts = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
    data = _ask_json(judge, _FAITHFULNESS_PROMPT, f"Excerpts:\n{excerpts}\n\nAnswer:\n{answer}")
    claims = [c for c in data.get("claims", []) if isinstance(c, dict) and "supported" in c]
    if not claims:
        return None, []
    supported = [c for c in claims if c["supported"] is True]
    unsupported = [str(c.get("claim", "")) for c in claims if c["supported"] is not True]
    return len(supported) / len(claims), unsupported


def judge_correctness(judge, question: str, reference: str, answer: str) -> tuple[float | None, str]:
    data = _ask_json(
        judge,
        _CORRECTNESS_PROMPT,
        f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to grade:\n{answer}",
    )
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        return None, ""
    return min(max(score, 0.0), 1.0), str(data.get("reason", ""))


def score_answers(class_id: str, questions: list[dict], use_judge: bool = True, progress=None) -> dict:
    """Generate an answer for every question and score its citations, faithfulness, and correctness.

    progress, if given, is called as progress(done, total) after each question.
    """
    check_ollama()
    judge = topics.make_json_llm() if use_judge else None
    rows = []
    for i, q in enumerate(questions, start=1):
        chunks = pipeline.retrieve(class_id, q["question"], k=config.DEFAULT_TOP_K)
        result = llm.generate_answer(q["question"], chunks, [])
        cited = result.citations
        on_target = [c for c in cited if _relevant(c.chunk_id.split(":", 1)[0], c.page, q)]
        row = {
            "question": q["question"],
            "expected": _expected(q),
            "answer": result.text,
            "cited": "; ".join(f"{c.document} p. {c.page}" for c in cited),
            "has_inline_citations": bool(re.search(r"\[\d+\]", result.text)),
            "citation_precision": len(on_target) / len(cited) if cited else 0.0,
            "cited_correct_page": bool(on_target),
            "faithfulness": None,
            "unsupported_claims": "",
            "correctness": None,
            "correctness_reason": "",
        }
        if judge:
            faith, unsupported = judge_faithfulness(judge, result.text, chunks)
            row["faithfulness"] = faith
            row["unsupported_claims"] = " | ".join(unsupported)
            if q.get("reference_answer"):
                row["correctness"], row["correctness_reason"] = judge_correctness(
                    judge, q["question"], q["reference_answer"], result.text
                )
        rows.append(row)
        if progress:
            progress(i, len(questions))

    def _avg(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return float(mean(values)) if values else None

    faith_scores = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    summary = {
        "questions": len(rows),
        "citation_precision": _avg("citation_precision"),
        "cited_correct_page": _avg("cited_correct_page"),
        "inline_citation_rate": _avg("has_inline_citations"),
        "faithfulness": _avg("faithfulness"),
        "hallucination_rate": float(mean(f < 1 for f in faith_scores)) if faith_scores else None,
        "answer_accuracy": _avg("correctness"),
    }
    return {"summary": summary, "rows": rows}


# Topics

def _tokens(name: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOPWORDS}


def topics_match(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    return len(ta & tb) / len(ta | tb) >= 0.5


def score_topic_list(expected: list[str], extracted: list[str]) -> dict:
    found = [e for e in expected if any(topics_match(e, x) for x in extracted)]
    correct = [x for x in extracted if any(topics_match(x, e) for e in expected)]
    precision = len(correct) / len(extracted) if extracted else 0.0
    recall = len(found) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed": [e for e in expected if e not in found],
        "extra": [x for x in extracted if x not in correct],
    }


def score_topics(class_id: str) -> dict:
    rows = []
    for doc in db.list_documents(class_id):
        expected = db.get_topic_labels(doc["id"])
        if not expected:
            continue
        extracted = db.get_document_topics(doc["id"])
        scores = score_topic_list(expected, extracted)
        rows.append(
            {
                "document": doc["filename"],
                "expected": len(expected),
                "extracted": len(extracted),
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "missed": "; ".join(scores["missed"]),
                "extra": "; ".join(scores["extra"]),
            }
        )
    summary = {"documents": len(rows)}
    if rows:
        for key in ("precision", "recall", "f1"):
            summary[key] = float(mean(r[key] for r in rows))
    return {"summary": summary, "rows": rows}


# Saving

def save_results(name: str, results: dict, directory: Path | None = None) -> Path:
    """Write results as JSON plus a CSV of the per-question rows. Returns the JSON path."""
    directory = directory or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = directory / f"{stem}.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    rows = results.get("rows") or []
    if rows:
        with open(directory / f"{stem}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return json_path


def _print_summary(title: str, summary: dict) -> None:
    print(f"\n{title}")
    for key, value in summary.items():
        shown = f"{value:.3f}" if isinstance(value, float) else value
        print(f"  {key:<22} {shown}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Course Atlas on a class's labeled questions.")
    parser.add_argument("class_name")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reranker", choices=["off", "on", "both"], default="both")
    parser.add_argument("--answers", action="store_true", help="also generate and judge answers (slow)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge when scoring answers")
    args = parser.parse_args()

    db.init_db()
    match = [c for c in db.list_classes() if c["name"].lower() == args.class_name.lower()]
    if not match:
        raise SystemExit(f"No class named {args.class_name!r}.")
    class_id = match[0]["id"]
    questions = db.list_eval_questions(class_id)
    print(f"{len(questions)} labeled questions")

    if questions:
        modes = {"off": [False], "on": [True], "both": [False, True]}[args.reranker]
        for use in modes:
            label = "reranker_on" if use else "reranker_off"
            res = score_retrieval(class_id, questions, k=args.k, use_reranker=use)
            _print_summary(f"Retrieval ({label})", res["summary"])
            print(f"  saved to {save_results(f'retrieval_{label}', res)}")

    if args.answers and questions:
        res = score_answers(
            class_id, questions, use_judge=not args.no_judge, progress=lambda d, t: print(f"  answered {d}/{t}")
        )
        _print_summary("Answers", res["summary"])
        print(f"  saved to {save_results('answers', res)}")

    res = score_topics(class_id)
    if res["rows"]:
        _print_summary("Topics", res["summary"])
        print(f"  saved to {save_results('topics', res)}")


if __name__ == "__main__":
    main()
