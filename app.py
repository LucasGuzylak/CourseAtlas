import html
import shutil
from pathlib import Path

import streamlit as st

from courseatlas import db, evaluation, parsing, pipeline, topics
from courseatlas.schemas import Citation

UPLOAD_DIR = db.DB_PATH.parent / "uploads"
ALLOWED_TYPES = ["pdf", "pptx"]

st.set_page_config(page_title="Course Atlas", page_icon="📚", layout="wide")
db.init_db()
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("notes", [])
st.session_state.setdefault("dismissed_topics", set())


def format_citations(citations: list[Citation]) -> str:
    return "  \n".join(f"[{i}] {c.document}, p. {c.page}" for i, c in enumerate(citations, 1))


@st.cache_data(max_entries=300, show_spinner=False)
def page_image(path: str, page: int, mtime: float) -> bytes | None:
    return parsing.render_page_image(path, page, dpi=110)


@st.cache_data(max_entries=300, show_spinner=False)
def page_text(path: str, page: int, mtime: float) -> str:
    return parsing.extract_single_page(path, page).text


def show_page(doc_id: str, page: int, label: str) -> None:
    """Render one page as an image for PDFs, or as its text for slide decks."""
    st.caption(label)
    doc = db.get_document(doc_id)
    if not doc or not Path(doc["path"]).exists():
        st.caption("This document has been removed.")
        return
    path = doc["path"]
    mtime = Path(path).stat().st_mtime
    try:
        image = page_image(path, page, mtime)
        if image:
            st.image(image, width="stretch")
            return
        text = page_text(path, page, mtime)
    except ValueError as e:
        st.caption(str(e))
        return
    with st.container(border=True):
        st.text(text or "(No text on this slide)")


def show_pages(pages: list[tuple[str, int, str]], per_row: int = 3) -> None:
    for start in range(0, len(pages), per_row):
        cols = st.columns(per_row)
        for col, (doc_id, page, label) in zip(cols, pages[start : start + per_row]):
            with col:
                show_page(doc_id, page, label)


def cited_pages(citations: list[Citation]) -> list[tuple[str, int, str]]:
    seen = set()
    pages = []
    for i, c in enumerate(citations, 1):
        doc_id = c.chunk_id.split(":", 1)[0]
        if (doc_id, c.page) in seen:
            continue
        seen.add((doc_id, c.page))
        pages.append((doc_id, c.page, f"[{i}] {c.document}, p. {c.page}"))
    return pages


def remove_document(class_id: str, doc: dict) -> None:
    pipeline.delete_document(class_id, doc["id"])
    db.delete_document(doc["id"])
    db.delete_orphan_topics(class_id)
    Path(doc["path"]).unlink(missing_ok=True)


def find_topics(class_id: str, doc: dict) -> None:
    """Run topic extraction for one document and queue a note about how it went."""
    try:
        count = topics.extract_and_save(class_id, doc["id"], doc["path"])
    except Exception as e:
        st.session_state.notes.append(
            ("warning", f"Couldn't find topics in {doc['filename']}: {e}. Is Ollama running?")
        )
        return
    if count:
        st.session_state.notes.append(("success", f"Found {count} topics in {doc['filename']}."))
    else:
        st.session_state.notes.append(("info", f"No topics found in {doc['filename']}."))


def toggle_weak(topic_id: str) -> None:
    db.set_weak(topic_id, st.session_state[f"topic_{topic_id}"])


def save_eval_question(class_id: str, doc_id: str) -> None:
    ss = st.session_state
    pages_key = f"eval_pages_{doc_id}"
    question = ss.eval_question.strip()
    if not question or not ss[pages_key]:
        ss.notes.append(("warning", "Write a question and pick at least one page."))
        return
    db.add_eval_question(class_id, question, doc_id, ss[pages_key], ss.eval_reference)
    ss.eval_question = ""
    ss.eval_reference = ""
    ss[pages_key] = []


def save_topic_labels(doc_id: str) -> None:
    db.set_topic_labels(doc_id, st.session_state[f"labels_{doc_id}"].splitlines())


def rounded(row: dict) -> dict:
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}


def ask_about_topic(topic: dict) -> None:
    st.session_state.pending_prompt = (
        f"Explain {topic['name']}. I marked it as a topic I'm struggling with, so start from the basics."
    )
    st.session_state.dismissed_topics.add(topic["id"])
    st.session_state.view = "Chat"


def create_class() -> None:
    name = st.session_state.new_class_name.strip()
    if not name:
        return
    if any(c["name"].lower() == name.lower() for c in db.list_classes()):
        st.session_state.flash = ("error", f"A class named “{name}” already exists.")
        return
    st.session_state.class_id = db.create_class(name)
    st.session_state.new_class_name = ""


def delete_class(class_id: str) -> None:
    for doc in db.list_documents(class_id):
        pipeline.delete_document(class_id, doc["id"])
    db.delete_class(class_id)
    shutil.rmtree(UPLOAD_DIR / class_id, ignore_errors=True)
    st.session_state.pop("class_id", None)


# Sidebar

with st.sidebar:
    st.title("📚 Course Atlas")

    classes = db.list_classes()
    names = {c["id"]: c["name"] for c in classes}
    if st.session_state.get("class_id") not in names:
        st.session_state.pop("class_id", None)

    class_id = None
    if classes:
        class_id = st.selectbox("Class", options=list(names), format_func=names.get, key="class_id")

    with st.form("new_class", clear_on_submit=False, border=False):
        st.text_input("New class", key="new_class_name", placeholder="e.g. Computer Architecture")
        st.form_submit_button("Create class", on_click=create_class, width="stretch")

    if flash := st.session_state.pop("flash", None):
        getattr(st, flash[0])(flash[1])

    if class_id:
        st.divider()
        view = st.radio(
            "View",
            ["Chat", "Documents", "Topics", "Slide Inspector", "Evaluation"],
            horizontal=False,
            label_visibility="collapsed",
            key="view",
        )

        with st.popover("Delete class", width="stretch"):
            st.write(f"Delete **{names[class_id]}** with all its documents and chat history?")
            st.button("Yes, delete", type="primary", on_click=delete_class, args=(class_id,))

if not class_id:
    st.header("Welcome to Course Atlas")
    st.info("Create a class in the sidebar to get started.")
    st.stop()

st.header(names[class_id])
documents = db.list_documents(class_id)

for level, text in st.session_state.notes:
    getattr(st, level)(text)
st.session_state.notes = []


# Documents view

if view == "Documents":
    files = st.file_uploader(
        "Upload slides, syllabi, homework, or notes",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if files and st.button("Add to class", type="primary"):
        existing = {d["filename"] for d in documents}
        class_dir = UPLOAD_DIR / class_id
        class_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if f.name in existing:
                st.session_state.notes.append(("warning", f"Skipped {f.name}: already uploaded."))
                continue
            doc_id = db.new_id()
            path = class_dir / f"{doc_id}{Path(f.name).suffix.lower()}"
            path.write_bytes(f.getbuffer())
            try:
                with st.spinner(f"Processing {f.name}..."):
                    result = pipeline.ingest(class_id, doc_id, path, document_name=f.name)
                db.add_document(class_id, doc_id, f.name, path, result.num_pages)
            except Exception as e:
                path.unlink(missing_ok=True)
                st.session_state.notes.append(("error", f"Couldn't process {f.name}: {e}"))
                continue
            with st.spinner(f"Finding topics in {f.name}..."):
                find_topics(class_id, db.get_document(doc_id))

        st.session_state.uploader_key += 1
        st.rerun()

    st.subheader(f"Documents ({len(documents)})")
    if not documents:
        st.caption("No documents yet.")
    for doc in documents:
        col_name, col_pages, col_delete = st.columns([6, 2, 1])
        col_name.write(doc["filename"])
        col_pages.caption(f"{doc['num_pages']} pages" if doc["num_pages"] else "")
        col_delete.button("🗑️", key=f"del_{doc['id']}", help="Delete", on_click=remove_document, args=(class_id, doc))


# Topics view

elif view == "Topics":
    missing = db.documents_without_topics(class_id)
    if missing:
        st.info(f"{len(missing)} document(s) have no topics yet.")
        if st.button("Find topics"):
            for doc in missing:
                with st.spinner(f"Finding topics in {doc['filename']}..."):
                    find_topics(class_id, doc)
            st.rerun()

    class_topics = db.list_topics(class_id)
    if not class_topics:
        st.caption("No topics yet. Topics are pulled from documents when you upload them.")
        st.stop()

    st.subheader("Which topics do you feel weak in?")
    cols = st.columns(2)
    for i, t in enumerate(class_topics):
        cols[i % 2].checkbox(
            t["name"],
            value=t["is_weak"],
            key=f"topic_{t['id']}",
            on_change=toggle_weak,
            args=(t["id"],),
            help=f"Covered on {t['num_pages']} page(s)",
        )

    weak = db.get_weak_topics(class_id)
    st.subheader("Pages to review")
    if not weak:
        st.caption("Check the topics you're struggling with and the pages that cover them will show up here.")
    for t in weak:
        with st.expander(t["name"], expanded=True):
            topic_pages = db.get_topic_pages(t["id"])
            by_doc: dict[str, list[int]] = {}
            for row in topic_pages:
                by_doc.setdefault(row["document"], []).append(row["page"])
            for document, pages in by_doc.items():
                st.markdown(f"**{document}**: pages {', '.join(map(str, pages))}")
            col_explain, col_show = st.columns([1, 3])
            col_explain.button("Explain this in chat", key=f"explain_{t['id']}", on_click=ask_about_topic, args=(t,))
            if col_show.toggle("Show pages", key=f"show_topic_{t['id']}"):
                show_pages([(r["doc_id"], r["page"], f"{r['document']}, p. {r['page']}") for r in topic_pages])


# Evaluation view

elif view == "Evaluation":
    if not documents:
        st.info("Upload documents first, then label questions against them here.")
        st.stop()

    doc_map = {d["id"]: d for d in documents}
    doc_name = lambda did: doc_map[did]["filename"]
    questions = db.list_eval_questions(class_id)
    tab_questions, tab_labels, tab_results = st.tabs(["Questions", "Topic labels", "Results"])

    with tab_questions:
        st.caption(f"{len(questions)} labeled questions. Aim for 30 to 50 spread across your documents.")
        col_form, col_preview = st.columns(2, gap="large")
        with col_form:
            eval_doc = st.selectbox("Document", list(doc_map), format_func=doc_name, key="eval_doc")
            num_pages = max(int(doc_map[eval_doc]["num_pages"] or 1), 1)
            eval_pages = st.multiselect(
                "Page(s) that answer the question", list(range(1, num_pages + 1)), key=f"eval_pages_{eval_doc}"
            )
            st.text_area("Question", key="eval_question", placeholder="How does a carry look-ahead adder avoid ripple delay?")
            st.text_area(
                "Reference answer (optional)",
                key="eval_reference",
                help="Used to grade answer accuracy. Leave it empty to only score retrieval and citations.",
            )
            st.button("Save question", type="primary", on_click=save_eval_question, args=(class_id, eval_doc))
        with col_preview:
            if eval_pages:
                show_page(eval_doc, eval_pages[0], f"{doc_name(eval_doc)}, p. {eval_pages[0]}")
            else:
                st.caption("Pick a page to preview it here.")

        if questions:
            st.divider()
        for q in questions:
            col_q, col_del = st.columns([12, 1])
            details = f"{q['document']}, p. {', '.join(map(str, q['pages']))}"
            if q["reference_answer"]:
                details += " · has reference answer"
            col_q.markdown(f"**{q['question']}**  \n{details}")
            col_del.button("🗑️", key=f"del_eval_{q['id']}", help="Delete", on_click=db.delete_eval_question, args=(q["id"],))

    with tab_labels:
        st.caption("List the topics you'd expect for a document. They're compared with the topics the app extracted.")
        label_doc = st.selectbox("Document", list(doc_map), format_func=doc_name, key="label_doc")
        saved_labels = db.get_topic_labels(label_doc)
        extracted = db.get_document_topics(label_doc)
        col_labels, col_extracted = st.columns(2, gap="large")
        with col_labels:
            st.text_area(
                "Correct topics, one per line", value="\n".join(saved_labels), key=f"labels_{label_doc}", height=260
            )
            st.button("Save labels", on_click=save_topic_labels, args=(label_doc,))
        with col_extracted:
            st.markdown("**Extracted topics**")
            st.markdown("\n".join(f"- {name}" for name in extracted) if extracted else "None yet.")
        if saved_labels:
            scores = evaluation.score_topic_list(saved_labels, extracted)
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{scores['precision']:.0%}")
            c2.metric("Recall", f"{scores['recall']:.0%}")
            c3.metric("F1", f"{scores['f1']:.0%}")
            if scores["missed"]:
                st.caption("Missed: " + "; ".join(scores["missed"]))
            if scores["extra"]:
                st.caption("Extra: " + "; ".join(scores["extra"]))

    with tab_results:
        if not questions:
            st.caption("Label some questions first.")
        st.caption(f"Every run is also saved as JSON and CSV in {evaluation.RESULTS_DIR}.")

        st.markdown("#### Retrieval")
        compare = st.checkbox("Also run with the reranker (downloads about 1 GB the first time)")
        if st.button("Run retrieval evaluation", disabled=not questions):
            runs = {}
            with st.spinner("Searching for every question..."):
                runs["reranker off"] = evaluation.score_retrieval(class_id, questions)
                if compare:
                    runs["reranker on"] = evaluation.score_retrieval(class_id, questions, use_reranker=True)
            for label, res in runs.items():
                evaluation.save_results(f"retrieval_{label.replace(' ', '_')}", res)
            st.session_state[f"eval_retrieval_{class_id}"] = runs
        if runs := st.session_state.get(f"eval_retrieval_{class_id}"):
            st.dataframe([rounded({"setup": label, **res["summary"]}) for label, res in runs.items()], hide_index=True)
            for label, res in runs.items():
                with st.expander(f"Per question ({label})"):
                    st.dataframe(res["rows"], hide_index=True)

        st.markdown("#### Answers")
        st.caption("Answers every question and has the LLM judge citations, faithfulness, and accuracy. Roughly a minute per question.")
        if st.button("Run answer evaluation", disabled=not questions):
            bar = st.progress(0.0, text="Starting...")
            try:
                res = evaluation.score_answers(
                    class_id, questions, progress=lambda done, total: bar.progress(done / total, text=f"{done}/{total} answered")
                )
            except RuntimeError as e:
                st.error(str(e))
            else:
                evaluation.save_results("answers", res)
                st.session_state[f"eval_answers_{class_id}"] = res
        if res := st.session_state.get(f"eval_answers_{class_id}"):
            st.dataframe([rounded(res["summary"])], hide_index=True)
            with st.expander("Per question"):
                st.dataframe([rounded(r) for r in res["rows"]], hide_index=True)

        st.markdown("#### Topics")
        topic_res = evaluation.score_topics(class_id)
        if topic_res["rows"]:
            st.dataframe([rounded(topic_res["summary"])], hide_index=True)
            st.dataframe([rounded(r) for r in topic_res["rows"]], hide_index=True)
        else:
            st.caption("Add topic labels for at least one document to score topic extraction.")


# Slide Inspector view (MVP 3-step pipeline visualization)

elif view == "Slide Inspector":
    st.caption("Slide image → Extracted text → Chunk card")
    if not documents:
        st.info("Upload a PDF or PPTX under **Documents** to inspect how slides are parsed and chunked.")
        st.stop()

    doc_map = {d["id"]: d for d in documents}
    col_doc, col_page = st.columns([3, 1])
    selected_doc_id = col_doc.selectbox(
        "Document",
        options=list(doc_map),
        format_func=lambda did: doc_map[did]["filename"],
    )
    selected_doc = doc_map[selected_doc_id]
    max_pages = max(int(selected_doc["num_pages"] or 1), 1)
    selected_page = int(
        col_page.number_input(
            "Slide / Page",
            min_value=1,
            max_value=max_pages,
            value=1,
            step=1,
        )
    )

    try:
        inspection = pipeline.inspect_page(
            class_id=class_id,
            doc_id=selected_doc["id"],
            file_path=selected_doc["path"],
            page=selected_page,
            document_name=selected_doc["filename"],
        )
    except Exception as e:
        st.error(f"Could not inspect page {selected_page}: {e}")
        st.stop()

    col_img, col_text, col_chunk = st.columns([1.15, 1.0, 1.0], gap="medium")

    with col_img:
        st.markdown("#### 1. Slide image")
        if inspection.image_bytes:
            st.image(inspection.image_bytes, width="stretch")
        else:
            st.info(
                "Slide image preview is rendered automatically for PDF files. "
                "Convert PowerPoint decks to PDF for slide thumbnails."
            )

    with col_text:
        st.markdown("#### 2. Extracted text")
        raw_text = inspection.extracted_text or "(No extractable text on this page)"
        escaped_text = html.escape(raw_text)
        st.markdown(
            f"""
            <div style="
                background-color: #f3f4f6;
                color: #111827;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 14px 16px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 0.85rem;
                line-height: 1.5;
                white-space: pre-wrap;
                overflow-x: auto;
            ">{escaped_text}</div>
            """,
            unsafe_allow_html=True,
        )

    with col_chunk:
        st.markdown("#### 3. Chunk card")
        if not inspection.chunks:
            st.caption("No chunks stored in ChromaDB for this page.")
        else:
            class_name = html.escape(names[class_id])
            for chunk in inspection.chunks:
                preview = " ".join(chunk.text.split())
                if len(preview) > 220:
                    preview = preview[:217].rstrip() + "..."
                escaped_doc = html.escape(chunk.document)
                escaped_preview = html.escape(preview)
                escaped_cid = html.escape(chunk.chunk_id)
                st.markdown(
                    f"""
                    <div style="
                        background-color: #ffffff;
                        color: #111827;
                        border: 1px solid #d1d5db;
                        border-left: 4px solid #2563eb;
                        border-radius: 8px;
                        padding: 14px 16px;
                        margin-bottom: 12px;
                        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
                    ">
                        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">
                            {escaped_doc} &middot; page {chunk.page}
                        </div>
                        <div style="font-size: 0.85rem; color: #4b5563; margin-bottom: 10px;">
                            Class: {class_name}
                        </div>
                        <div style="font-size: 0.9rem; color: #1f2937; font-style: italic; line-height: 1.45;">
                            &ldquo;{escaped_preview}&rdquo;
                        </div>
                        <div style="font-size: 0.75rem; color: #6b7280; margin-top: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;">
                            chunk_id: {escaped_cid}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# Chat view

else:
    history = db.get_history(class_id)

    if not documents:
        st.info("No documents in this class yet. Upload some under Documents for grounded answers.")

    if history:
        st.button("Clear chat", on_click=db.clear_history, args=(class_id,))

    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["citations"]:
                st.caption(format_citations(msg["citations"]))
                if st.toggle("Show cited pages", key=f"cited_{class_id}_{i}"):
                    show_pages(cited_pages(msg["citations"]))

    weak = db.get_weak_topics(class_id)
    offer = next((t for t in weak if t["id"] not in st.session_state.dismissed_topics), None)
    if offer and "pending_prompt" not in st.session_state:
        which = "the first topic" if offer["id"] == weak[0]["id"] else "another topic"
        with st.chat_message("assistant"):
            st.markdown(f"Would you like me to explain **{offer['name']}**, {which} you marked as confusing?")
            col_yes, col_no, _ = st.columns([1, 1, 4])
            col_yes.button("Yes, explain it", key="offer_yes", on_click=ask_about_topic, args=(offer,))
            col_no.button("Not now", key="offer_no", on_click=st.session_state.dismissed_topics.add, args=(offer["id"],))

    prompt = st.chat_input(f"Ask about {names[class_id]}") or st.session_state.pop("pending_prompt", None)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        db.add_message(class_id, "user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.answer(
                    class_id, prompt, [{"role": m["role"], "content": m["content"]} for m in history]
                )
            st.markdown(result.text)
            if result.citations:
                st.caption(format_citations(result.citations))
        db.add_message(class_id, "assistant", result.text, result.citations)
        st.rerun()
