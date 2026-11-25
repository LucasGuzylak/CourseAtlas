import shutil
from pathlib import Path

import streamlit as st

from courseatlas import db, pipeline
from courseatlas.schemas import Citation

UPLOAD_DIR = db.DB_PATH.parent / "uploads"
ALLOWED_TYPES = ["pdf", "pptx"]

st.set_page_config(page_title="Course Atlas", page_icon="📚", layout="wide")
db.init_db()
st.session_state.setdefault("uploader_key", 0)


def format_citations(citations: list[Citation]) -> str:
    return "  \n".join(f"[{i}] {c.document}, p. {c.page}" for i, c in enumerate(citations, 1))


def remove_document(class_id: str, doc: dict) -> None:
    pipeline.delete_document(class_id, doc["id"])
    db.delete_document(doc["id"])
    Path(doc["path"]).unlink(missing_ok=True)


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
        view = st.radio("View", ["Chat", "Documents"], horizontal=True, label_visibility="collapsed")

        with st.popover("Delete class", width="stretch"):
            st.write(f"Delete **{names[class_id]}** with all its documents and chat history?")
            st.button("Yes, delete", type="primary", on_click=delete_class, args=(class_id,))

if not class_id:
    st.header("Welcome to Course Atlas")
    st.info("Create a class in the sidebar to get started.")
    st.stop()

st.header(names[class_id])
documents = db.list_documents(class_id)


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
                st.warning(f"Skipped {f.name}: already uploaded.")
                continue
            doc_id = db.new_id()
            path = class_dir / f"{doc_id}{Path(f.name).suffix.lower()}"
            path.write_bytes(f.getbuffer())
            try:
                with st.spinner(f"Processing {f.name}..."):
                    result = pipeline.ingest(class_id, doc_id, path)
                db.add_document(class_id, doc_id, f.name, path, result.num_pages)
            except Exception as e:
                path.unlink(missing_ok=True)
                st.error(f"Couldn't process {f.name}: {e}")

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


# Chat view

else:
    history = db.get_history(class_id)

    if not documents:
        st.info("No documents in this class yet. Upload some under Documents for grounded answers.")

    if history:
        st.button("Clear chat", on_click=db.clear_history, args=(class_id,))

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["citations"]:
                st.caption(format_citations(msg["citations"]))

    if prompt := st.chat_input(f"Ask about {names[class_id]}"):
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
