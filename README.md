# Course Atlas

Course Atlas is a study assistant built around a student's own course material. Students upload slides, syllabi, homework, and notes for each class, then ask questions and get answers that cite the specific slide or page they're based on. It will also let students mark the topics they're struggling with and surface the pages that cover them.

Answers are generated with retrieval-augmented generation (RAG). Each question is matched against the class's uploaded material, and a local LLM writes a response grounded in the pages it finds.

## Current status

The app works from upload to chat, but the pipeline behind it is a placeholder that returns fake answers and citations. The UI and storage are being built against that placeholder so they're ready when the real pipeline is.

Done:
- Shared data types (`schemas.py`)
- SQLite storage for classes, documents, chat history, and topics (`db.py`)
- Streamlit app with class management, uploads, chat, and citations (`app.py`)

Not started:
- Real pipeline: parsing, embeddings, ChromaDB, LLM
- Weak-topic flow
- Page previews
- Evaluation

## Setup

You'll need [uv](https://docs.astral.sh/uv/). The project is pinned to Python 3.11, and uv will download it if you don't have it.

```bash
git clone https://github.com/LucasGuzylak/CourseAtlas.git
cd CourseAtlas
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Activate the environment in every new terminal before working on the project.

## Running the app

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

## Project layout

```
app.py                  Streamlit interface
requirements.txt
courseatlas/
    schemas.py          data types shared between the app and the pipeline
    pipeline.py         document processing and question answering (placeholder)
    db.py               SQLite storage
```

The app writes its database and uploaded files to `data/`, which is ignored by git. Uploads are stored under `data/uploads/<class_id>/`. To use a different database file, set the `COURSEATLAS_DB` environment variable.

## How the pieces connect

`app.py` handles the interface and does no real work on its own. It reads and writes app data through `db.py` and sends documents and questions to `pipeline.py`.

Storage is split in two. SQLite holds classes, documents, chat history, and topics. ChromaDB will hold only the embeddings. The two are linked by `doc_id`.

## Pipeline interface

The app calls the pipeline through four functions. The real implementation needs to keep these signatures and return types so the app keeps working without changes.

```python
ingest(class_id, doc_id, file_path) -> IngestResult
retrieve(class_id, query, k=5) -> list[Chunk]
answer(class_id, query, history) -> AnswerResult
delete_document(class_id, doc_id) -> None
```

Rules both sides rely on:

- The app generates `doc_id` and saves the file before calling `ingest()`.
- `retrieve()` and `answer()` only search documents belonging to the given class.
- `history` contains the earlier turns of the conversation, not the current question.
- `delete_document()` removes the document from the vector store only. The app handles the database row and the file.
- Citation markers in the answer text (`[1]`, `[2]`) refer to entries in `citations`, in order.

The data types live in `courseatlas/schemas.py`. Page numbers are 1-based, and chunk IDs follow the format `doc_id:page:n`.

## Database

| Table | Contents |
|---|---|
| `classes` | Class names (unique) |
| `documents` | Uploaded files, their class, storage path, and page count |
| `chat_messages` | Chat history per class, including citations |
| `topics` | Topics per class and whether the student marked them as weak |
| `chunk_topics` | Which pages belong to which topics |

Deleting a class also deletes its documents, messages, and topics. Deleting a document removes its topic links. The app calls `db.init_db()` on startup, which creates any missing tables and is safe to run repeatedly.

## Planned stack

| Layer | Choice |
|---|---|
| Interface | Streamlit |
| PDF parsing | PyMuPDF |
| Slide parsing | python-pptx |
| Chunking | One chunk per page or slide, with long pages split by LangChain's `RecursiveCharacterTextSplitter` |
| Embeddings | `BAAI/bge-small-en-v1.5` via Sentence Transformers |
| Vector store | ChromaDB, one persistent collection filtered by class |
| LLM | `qwen3:8b` or `llama3.1:8b` running locally through Ollama |
| Orchestration | LangChain, used sparingly |
| App data | SQLite |
| Reranking (stretch goal) | `BAAI/bge-reranker-base` |
| Evaluation | A hand-labeled question set for retrieval and citation accuracy, plus RAGAS for faithfulness |

## Responsibilities

**Engineer 1** owns the backend and AI pipeline: parsing, chunking, embeddings, the vector store, retrieval, answer generation, and citations.

**Engineer 2 (Guzylak)** owns the application: the Streamlit interface, class and document management, persistent storage, topic extraction and weak-topic selection, page previews, and usability testing.

## Contributions

### Guzylak

- Set up the project environment: Python 3.11 pinned with uv, a project virtual environment, and a `.gitignore` that keeps local data and environments out of the repo.
- Defined the shared data types in `schemas.py` that the app and pipeline use to exchange chunks, citations, answers, and chat messages.
- Wrote the pipeline interface in `pipeline.py` as a placeholder that returns fake data, so the app could be built before the real pipeline exists.
- Built the SQLite storage layer in `db.py`: tables for classes, documents, chat history, topics, and topic-to-page links, with cascading deletes and helper functions for each.
- Built the Streamlit app in `app.py`: creating, switching, and deleting classes; uploading and removing documents; per-class chat with saved history; and citations shown under each answer.
- Wrote this README.

## Next steps

1. Weak-topic flow
2. Page previews for cited pages
3. Evaluation setup: a usefulness survey and a labeled question set
4. Replace the placeholder pipeline with the real implementation
