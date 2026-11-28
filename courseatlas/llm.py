import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from courseatlas import config
from courseatlas.schemas import AnswerResult, Chunk, Citation, Message

_SYSTEM_PROMPT = """You are Course Atlas, a helpful academic study assistant.
Answer the student's question using ONLY the provided course material excerpts below.

Rules:
1. Ground every claim in the provided excerpts and cite sources inline with bracketed numbers like [1] or [2] that match the excerpt numbers.
2. You may cite multiple sources like [1][2] when combining information.
3. Do not invent facts, formulas, or page numbers that are not supported by the excerpts.
4. If the excerpts do not contain enough information to answer the question, clearly say so and briefly mention what the excerpts do cover.
5. Keep explanations clear, structured, and student-friendly."""

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _format_context(chunks: list[Chunk]) -> str:
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{idx}] Source: {chunk.document} (page {chunk.page})\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def _build_messages(query: str, chunks: list[Chunk], history: list[Message]) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]

    recent_history = history[-(config.MAX_HISTORY_TURNS * 2) :] if config.MAX_HISTORY_TURNS > 0 else []
    for turn in recent_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    context_block = _format_context(chunks)
    user_prompt = (
        f"Course material excerpts:\n\n{context_block}\n\n"
        f"Student question: {query}\n\n"
        "Answer with inline citations [1], [2], etc.:"
    )
    messages.append(HumanMessage(content=user_prompt))
    return messages


def normalize_citations(text: str, chunks: list[Chunk]) -> tuple[str, list[Citation]]:
    """Renumber valid inline [n] citations in order of first appearance so [i] maps to citations[i-1]."""
    cleaned = _THINK_TAG_RE.sub("", text).strip()
    valid_range = range(1, len(chunks) + 1)

    old_to_new: dict[int, int] = {}
    ordered_citations: list[Citation] = []

    def _replace(match: re.Match[str]) -> str:
        num = int(match.group(1))
        if num not in valid_range:
            return ""
        if num not in old_to_new:
            old_to_new[num] = len(ordered_citations) + 1
            chunk = chunks[num - 1]
            ordered_citations.append(
                Citation(chunk_id=chunk.chunk_id, document=chunk.document, page=chunk.page)
            )
        return f"[{old_to_new[num]}]"

    normalized_text = _CITATION_RE.sub(_replace, cleaned)
    # Clean up any extra spaces left if an out-of-range marker was stripped
    normalized_text = re.sub(r" {2,}", " ", normalized_text)
    normalized_text = re.sub(r" +([.,;:!?])", r"\1", normalized_text).strip()

    if not ordered_citations and chunks:
        # Fallback: attach the top retrieved chunks so the student still sees sources
        ordered_citations = [
            Citation(chunk_id=c.chunk_id, document=c.document, page=c.page) for c in chunks
        ]

    return normalized_text, ordered_citations


def _fallback_answer(chunks: list[Chunk], error: Exception) -> AnswerResult:
    """Return retrieved passages with citations when Ollama is unavailable."""
    citations = [Citation(chunk_id=c.chunk_id, document=c.document, page=c.page) for c in chunks]
    excerpts = []
    for idx, chunk in enumerate(chunks, start=1):
        snippet = chunk.text.replace("\n", " ").strip()
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."
        excerpts.append(f"- **[{idx}] {chunk.document}, p. {chunk.page}**: {snippet}")

    text = (
        f"Could not reach the local Ollama LLM (`{config.OLLAMA_MODEL}` at `{config.OLLAMA_BASE_URL}`): "
        f"`{error}`.\n\n"
        f"Make sure Ollama is running and the model is pulled (`ollama pull {config.OLLAMA_MODEL}`). "
        "Here are the most relevant excerpts retrieved from your course materials:\n\n"
        + "\n".join(excerpts)
    )
    return AnswerResult(text=text, citations=citations)


def generate_answer(query: str, chunks: list[Chunk], history: list[Message]) -> AnswerResult:
    """Generate a citation-grounded answer using the configured Ollama LLM."""
    if not chunks:
        return AnswerResult(
            text="I couldn't find any relevant course materials for this class yet. "
            "Upload slides, notes, or syllabi under **Documents** and try asking again.",
            citations=[],
        )

    messages = _build_messages(query, chunks, history)
    try:
        llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )
        response = llm.invoke(messages)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        return _fallback_answer(chunks, exc)

    answer_text, citations = normalize_citations(raw_text, chunks)
    return AnswerResult(text=answer_text, citations=citations)
