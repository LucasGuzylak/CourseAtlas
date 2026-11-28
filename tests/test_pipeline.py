import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
from pptx import Presentation

from courseatlas import chunking, db, llm, parsing, pipeline, vectorstore
from courseatlas.schemas import Chunk


def _make_sample_pdf(path: Path, slides: list[tuple[str, list[str]]]) -> None:
    doc = pymupdf.open()
    for title, bullets in slides:
        page = doc.new_page(width=720, height=405)  # 16:9 slide aspect ratio
        page.insert_text((48, 64), title, fontsize=24, fontname="helv")
        y = 118
        for bullet in bullets:
            page.insert_text((56, y), f"• {bullet}", fontsize=15, fontname="helv")
            y += 32
    doc.save(str(path))
    doc.close()


def _make_sample_pptx(path: Path, slides: list[tuple[str, list[str]]]) -> None:
    prs = Presentation()
    layout = prs.slide_layouts[1]
    for title, bullets in slides:
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = title
        tf = slide.placeholders[1].text_frame
        tf.text = bullets[0]
        for b in bullets[1:]:
            p = tf.add_paragraph()
            p.text = b
    prs.save(str(path))


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        db.DB_PATH = self.tmp_path / "test.db"
        os.environ["COURSEATLAS_CHROMA_DIR"] = str(self.tmp_path / "chroma")
        vectorstore._clients.clear()
        db.init_db()

    def tearDown(self) -> None:
        vectorstore._clients.clear()
        os.environ.pop("COURSEATLAS_CHROMA_DIR", None)
        self.tmp.cleanup()

    def test_pdf_and_pptx_parsing_and_chunking(self) -> None:
        pdf_path = self.tmp_path / "Lecture_03.pdf"
        _make_sample_pdf(
            pdf_path,
            [
                ("Course Overview", ["Number representation", "Binary arithmetic"]),
                (
                    "Carry Look-Ahead Adders",
                    [
                        "Carry look-ahead adders compute carries in parallel using generate and propagate signals.",
                        "Gi = Ai AND Bi, Pi = Ai XOR Bi",
                        "Reduces addition delay from O(n) to O(log n).",
                    ],
                ),
            ],
        )
        pages = parsing.extract_pages(pdf_path)
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[1].page, 2)
        self.assertIn("Carry look-ahead adders compute carries in parallel", pages[1].text)

        img = parsing.render_page_image(pdf_path, 2)
        self.assertIsNotNone(img)
        self.assertTrue(img.startswith(b"\x89PNG"))

        pptx_path = self.tmp_path / "Lecture_04.pptx"
        _make_sample_pptx(
            pptx_path,
            [("Pipelining", ["Instruction fetch", "Decode", "Execute"])],
        )
        pptx_pages = parsing.extract_pages(pptx_path)
        self.assertEqual(len(pptx_pages), 1)
        self.assertIn("Pipelining", pptx_pages[0].text)
        self.assertIn("Instruction fetch", pptx_pages[0].text)

        # Long page splitting preserves 1-based page number and sequential chunk_id
        long_page = parsing.PageContent(page=5, text=("Carry look-ahead logic. " * 120).strip())
        chunks = chunking.chunk_pages("cls1", "doc1", "Lecture_03.pdf", [long_page], chunk_size=300, chunk_overlap=40)
        self.assertGreater(len(chunks), 1)
        for idx, c in enumerate(chunks):
            self.assertEqual(c.chunk_id, f"doc1:5:{idx}")
            self.assertEqual(c.page, 5)

    def test_ingest_retrieve_isolation_inspect_and_delete(self) -> None:
        arch_id = db.create_class("Computer Architecture")
        os_id = db.create_class("Operating Systems")

        arch_pdf = self.tmp_path / "arch_doc.pdf"
        _make_sample_pdf(
            arch_pdf,
            [
                ("Ripple-Carry Adders", ["Each full adder waits for the previous carry bit."]),
                (
                    "Carry Look-Ahead Adders",
                    [
                        "Carry look-ahead adders compute carries in parallel.",
                        "Uses generate (G) and propagate (P) signals to avoid ripple delay.",
                    ],
                ),
            ],
        )
        os_pdf = self.tmp_path / "os_doc.pdf"
        _make_sample_pdf(
            os_pdf,
            [
                (
                    "Virtual Memory and Paging",
                    ["Page tables map virtual addresses to physical frames.", "TLB caches recent translations."],
                )
            ],
        )

        arch_doc_id = db.new_id()
        res_arch = pipeline.ingest(arch_id, arch_doc_id, arch_pdf)
        db.add_document(arch_id, arch_doc_id, "Lecture_03.pdf", arch_pdf, res_arch.num_pages)
        self.assertEqual(res_arch.num_pages, 2)
        self.assertEqual(res_arch.num_chunks, 2)

        os_doc_id = db.new_id()
        res_os = pipeline.ingest(os_id, os_doc_id, os_pdf, document_name="OS_Week05.pdf")
        db.add_document(os_id, os_doc_id, "OS_Week05.pdf", os_pdf, res_os.num_pages)
        self.assertEqual(res_os.num_pages, 1)

        # Retrieval in Computer Architecture finds Carry Look-Ahead Adders on page 2 with original filename
        hits = pipeline.retrieve(arch_id, "How do carry look-ahead adders work?", k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].class_id, arch_id)
        self.assertEqual(hits[0].document, "Lecture_03.pdf")
        self.assertEqual(hits[0].page, 2)

        # Strict class_id isolation: querying OS class never returns Architecture chunks
        os_hits = pipeline.retrieve(os_id, "How do carry look-ahead adders work?", k=5)
        self.assertEqual(len(os_hits), 1)
        self.assertTrue(all(h.class_id == os_id for h in os_hits))
        self.assertEqual(os_hits[0].document, "OS_Week05.pdf")

        # 3-step MVP page inspection
        insp = pipeline.inspect_page(arch_id, arch_doc_id, arch_pdf, page=2, document_name="Lecture_03.pdf")
        self.assertIsNotNone(insp.image_bytes)
        self.assertIn("Carry look-ahead adders compute carries in parallel", insp.extracted_text)
        self.assertEqual(len(insp.chunks), 1)
        self.assertEqual(insp.chunks[0].document, "Lecture_03.pdf")
        self.assertEqual(insp.chunks[0].page, 2)

        # Answer with mocked ChatOllama verifies inline citation renumbering
        fake_resp = MagicMock()
        fake_resp.content = (
            "<think>checking slides</think>"
            "Carry look-ahead adders compute carries in parallel using G and P signals [1], "
            "unlike ripple-carry adders that wait for the previous bit [2]."
        )
        with patch("courseatlas.llm.ChatOllama") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = fake_resp
            ans = pipeline.answer(arch_id, "How do carry look-ahead adders work?", history=[])
        self.assertNotIn("<think>", ans.text)
        self.assertIn("[1]", ans.text)
        self.assertEqual(len(ans.citations), 2)
        self.assertEqual(ans.citations[0].document, "Lecture_03.pdf")
        self.assertEqual(ans.citations[0].page, 2)

        # Deletion removes vectors for that document only
        pipeline.delete_document(arch_id, arch_doc_id)
        self.assertEqual(pipeline.retrieve(arch_id, "carry look-ahead", k=5), [])
        self.assertEqual(len(pipeline.retrieve(os_id, "virtual memory", k=5)), 1)

    def test_citation_normalization(self) -> None:
        chunks = [
            Chunk("d:1:0", "c", "d", "Lec1.pdf", 1, "first"),
            Chunk("d:2:0", "c", "d", "Lec1.pdf", 2, "second"),
            Chunk("d:3:0", "c", "d", "Lec1.pdf", 3, "third"),
        ]
        text, cites = llm.normalize_citations(
            "<think>internal</think>Point A [3] and point B [1] and bogus [9].", chunks
        )
        self.assertEqual(text, "Point A [1] and point B [2] and bogus.")
        self.assertEqual([(c.document, c.page) for c in cites], [("Lec1.pdf", 3), ("Lec1.pdf", 1)])


if __name__ == "__main__":
    unittest.main()
