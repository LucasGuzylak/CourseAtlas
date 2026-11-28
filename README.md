# Course Atlas

Course Atlas is a study assistant built around a student's own course material. Students upload slides, syllabi, homework, and notes for each class, then ask questions and get answers that cite the specific slide or page they're based on. The app also pulls the topics out of each upload, lets students mark the ones they're struggling with, and shows the pages that cover them.

Answers are generated with retrieval-augmented generation (RAG). Each question is matched against the class's uploaded material, and a local LLM writes a response grounded in the pages it finds.

## Current status

The app works from upload to chat, and the pipeline behind it is now the real one.

Done:
- Shared data types and SQLite storage
- Streamlit app with class management, uploads, chat, and citations
- Real RAG pipeline: PDF and PowerPoint parsing, page-level chunking, embeddings, ChromaDB, and answers from a local LLM through Ollama
- Optional reranker
- Slide Inspector showing how a page is parsed and chunked
- Topic extraction on upload, weak-topic selection, and a follow-up offer in chat
- Previews of cited pages and of the pages behind each weak topic
- Evaluation tools: labeled questions, retrieval and citation metrics, an LLM judge for faithfulness and accuracy, topic accuracy, and a usability survey

Remaining:
- Labeling 30 to 50 questions on real lecture decks and running the full evaluation
- Usability testing with students

## Setup

You'll need [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com). The project is pinned to Python 3.11, and uv will download it if you don't have it.

```bash
git clone https://github.com/LucasGuzylak/CourseAtlas.git
cd CourseAtlas
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
ollama pull qwen3:8b
```

Activate the environment in every new terminal before working on the project. The first upload downloads the embedding model (about 130 MB).

## Running the app

Start Ollama first, either by opening the Ollama app or with:

```bash
ollama serve
```

Then, in another terminal:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. Uploads and search still work without Ollama, but chat falls back to showing the retrieved excerpts and topic extraction is skipped. You can run it later from the Topics view.

## Using the app

Each class has five views in the sidebar:

- **Chat**: ask questions about the class's documents. Every answer lists its citations, and "Show cited pages" displays the pages themselves. If you've marked weak topics, the chat offers to explain the first one.
- **Documents**: upload PDF and PowerPoint files, or delete them. Topics are extracted right after each upload.
- **Topics**: check the topics you feel weak in. Each weak topic lists the pages that cover it, with an option to show them or ask about it in chat.
- **Slide Inspector**: pick a page and see the slide image, the extracted text, and the chunks stored for it.
- **Evaluation**: label questions and expected topics, and run the evaluation.

PDF pages are shown as images. PowerPoint slides are shown as their extracted text. Convert a deck to PDF if you want slide images.

## Evaluation

The Evaluation view has three tabs:

- **Questions**: pick a document and the page or pages that answer a question, then write the question and, optionally, a reference answer.
- **Topic labels**: list the topics you'd expect for each document. They're compared with the topics the app extracted.
- **Results**: run the evaluations and see the scores.

What gets measured:

| Measure | How |
|---|---|
| Retrieval relevance | hit@1, hit@3, hit@5, and MRR against the labeled pages, with the reranker off and on |
| Citation correctness | Share of cited pages that are labeled pages, and whether any citation hit a labeled page |
| Hallucination rate | The local LLM splits each answer into claims and checks them against the retrieved pages. An answer with any unsupported claim counts as a hallucination. |
| Answer accuracy | The local LLM grades each answer against the reference answer (1, 0.5, or 0) |
| Topic identification | Precision, recall, and F1 of extracted topics against the labels |
| Student-rated usefulness | Google Form built from [docs/usability_survey.md](docs/usability_survey.md) |

Answering and judging takes about a minute per question, so long runs are easier from the terminal:

```bash
python -m courseatlas.evaluation "Computer Architecture" --answers
```

Leave out `--answers` to run only the retrieval and topic checks, which take seconds. Every run is saved as JSON and CSV in `data/eval/`.

## Project layout

```
app.py                      Streamlit interface
requirements.txt
courseatlas/
    schemas.py              data types shared between the app and the pipeline
    db.py                   SQLite storage
    config.py               settings, overridable with environment variables
    parsing.py              text extraction and page rendering for PDF and PPTX
    chunking.py             one chunk per page, with long pages split
    embeddings.py           embeddings and the optional reranker
    vectorstore.py          ChromaDB storage and class-filtered search
    llm.py                  answer generation and citation handling
    pipeline.py             ingest, retrieve, answer, delete, and page inspection
    topics.py               topic extraction and saving
    evaluation.py           evaluation metrics, LLM judge, and command-line runner
tests/
    test_pipeline.py
    test_topics.py
    test_evaluation.py
docs/
    usability_survey.md     questions for the usability survey
```

The app writes its database, uploaded files, ChromaDB index, and evaluation results to `data/`, which is ignored by git. Uploads are stored under `data/uploads/<class_id>/`.

Run the tests with:

```bash
python -m unittest discover -s tests
```

The tests don't need Ollama. The LLM is mocked.

## How the pieces connect

`app.py` handles the interface. It reads and writes app data through `db.py`, sends documents and questions to `pipeline.py`, and calls `topics.py` after each upload.

Storage is split in two. SQLite holds classes, documents, chat history, topics, and evaluation labels. ChromaDB holds the chunks and their embeddings. The two are linked by `doc_id`, and topics link to pages by chunk ID.

When a document is uploaded:

1. The app saves the file and generates a `doc_id`.
2. `pipeline.ingest()` extracts the text of each page, chunks it, embeds the chunks, and stores them in ChromaDB.
3. The app adds the document to SQLite.
4. `topics.extract_and_save()` asks the LLM for the document's topics and links each one to its pages.

When a question is asked, `pipeline.answer()` embeds it, retrieves the top five chunks from that class only, and has the LLM write an answer with numbered citations.

## Pipeline interface

The app calls the pipeline through these functions:

```python
ingest(class_id, doc_id, file_path, document_name=None) -> IngestResult
retrieve(class_id, query, k=5) -> list[Chunk]
answer(class_id, query, history) -> AnswerResult
delete_document(class_id, doc_id) -> None
inspect_page(class_id, doc_id, file_path, page, document_name=None) -> PageInspection
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
| `topics` | Topics per class, whether the student marked them as weak, and the order they were marked |
| `chunk_topics` | Which pages belong to which topics |
| `eval_questions` | Labeled evaluation questions with their correct pages and optional reference answers |
| `eval_topic_labels` | Expected topics per document, for scoring topic extraction |

Deleting a class also deletes its documents, messages, topics, and evaluation labels. Deleting a document removes its topic links, its evaluation labels, and any topics left without pages. The app calls `db.init_db()` on startup, which creates any missing tables and columns and is safe to run repeatedly.

## Configuration

Settings live in `courseatlas/config.py` and can be overridden with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `COURSEATLAS_DB` | `data/courseatlas.db` | SQLite database file |
| `COURSEATLAS_CHROMA_DIR` | `data/chroma` | ChromaDB directory |
| `COURSEATLAS_LLM_MODEL` | `qwen3:8b` | Ollama model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `COURSEATLAS_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `COURSEATLAS_USE_RERANKER` | `0` | Set to `1` to rerank results with `BAAI/bge-reranker-base` |
| `COURSEATLAS_TOP_K` | `5` | Chunks used per answer |
| `COURSEATLAS_CHUNK_SIZE` | `1200` | Characters before a page is split |

## Stack

| Layer | Choice |
|---|---|
| Interface | Streamlit |
| PDF parsing | PyMuPDF |
| Slide parsing | python-pptx |
| Chunking | One chunk per page or slide, with long pages split by LangChain's `RecursiveCharacterTextSplitter` |
| Embeddings | `BAAI/bge-small-en-v1.5` via Sentence Transformers |
| Vector store | ChromaDB, one persistent collection filtered by class |
| LLM | `qwen3:8b` running locally through Ollama |
| Orchestration | LangChain, used for the Ollama client and text splitting |
| App data | SQLite |
| Reranking | `BAAI/bge-reranker-base`, off by default |
| Evaluation | Hand-labeled questions for retrieval and citation metrics, plus the local LLM as a judge for faithfulness and accuracy |

## Responsibilities

**Engineer 1 (Ceurty)** owns the backend and AI pipeline: parsing, chunking, embeddings, the vector store, retrieval, answer generation, and citations.

**Engineer 2 (Guzylak)** owns the application: the Streamlit interface, class and document management, persistent storage, topic extraction and weak-topic selection, page previews, evaluation, and usability testing.

## Contributions

### Ceurty

- Wrote text extraction for PDFs (PyMuPDF) and PowerPoint decks (python-pptx), including text inside slide tables.
- Chunked by page, so every chunk comes from exactly one page and only long pages get split.
- Set up embeddings with `bge-small-en-v1.5`, using the query prefix the model expects, and added the optional `bge-reranker-base` reranker.
- Built the ChromaDB vector store. Every search is filtered by class, and each chunk keeps its document name and page number.
- Hooked answer generation up to Ollama. The prompt tells the model to stick to the retrieved excerpts. The citation code then strips qwen3's thinking text, drops markers that don't match an excerpt, and renumbers the rest in order.
- Added a fallback that shows the retrieved excerpts when Ollama isn't running.
- Made the Slide Inspector, which shows a page's image, extracted text, and stored chunks side by side.
- Put the settings in `config.py` so each one can be changed with an environment variable.
- Wrote the pipeline tests for parsing, chunking, class isolation, citations, and deletion.

### Guzylak

- Set up the project environment: Python 3.11 pinned with uv, a project virtual environment, and a `.gitignore` that keeps local data and environments out of the repo.
- Defined the shared data types in `schemas.py` that the app and pipeline use to exchange chunks, citations, answers, and chat messages.
- Wrote the pipeline interface in `pipeline.py` as a placeholder, so the app could be built before the real pipeline existed.
- Built the SQLite storage layer in `db.py`: tables for classes, documents, chat history, topics, topic-to-page links, and evaluation labels, with cascading deletes and helper functions for each.
- Built the Streamlit app in `app.py`: creating, switching, and deleting classes; uploading and removing documents; per-class chat with saved history; and citations shown under each answer.
- Wrote topic extraction in `topics.py`. The local LLM reads each upload in batches and sends back its topics and pages as JSON, and duplicates or sub-topics like "Carry Look-Ahead: Delay" get merged. Tuned the prompt on real qwen3 output until it covered every teaching slide and skipped title, agenda, and homework slides.
- Added the weak-topic flow. Students check off topics in the Topics view and see the pages that cover them, and the chat offers to explain the first one they marked.
- Put page previews under chat citations and weak topics. They only render when you turn them on, and they're cached.
- Set up the evaluation in `evaluation.py` and the Evaluation view. You label questions and expected topics, then get hit@k and MRR with the reranker on and off, citation accuracy, and topic precision and recall. An LLM judge scores faithfulness and answer accuracy. Long runs can go through the command line, and results are saved as JSON and CSV.
- Came up with the usability survey questions in `docs/usability_survey.md`.
- Wrote tests for topics and evaluation, and rewrote this README.

## Next steps

1. Upload two or three real lecture decks and label 30 to 50 questions
2. Run the full evaluation and compare the reranker on and off
3. Run usability testing with the survey
4. If there's time: OCR for slides that are only images, slide images for PowerPoint files, and RAGAS as a second faithfulness check
