"""Tests for PDF text extraction."""

import pytest

from src.rag.loader import load_pdf


def test_load_pdf_extracts_text_per_page(pdf_factory):
    path = pdf_factory("guide.pdf", [
        ["Vector search finds semantically similar chunks."],
        ["BM25 ranks chunks by keyword overlap."],
    ])

    pages = load_pdf(path)

    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "Vector search" in pages[0]["text"]
    assert pages[1]["page"] == 2
    assert "BM25" in pages[1]["text"]


def test_load_pdf_keeps_blank_pages(pdf_factory):
    path = pdf_factory("blank.pdf", [[], ["Only this page has text."]])

    pages = load_pdf(path)

    assert len(pages) == 2
    assert pages[0]["text"] == ""
    assert pages[1]["text"] == "Only this page has text."


def test_load_pdf_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_rejects_non_pdf_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Not a PDF file"):
        load_pdf(path)


def test_load_pdf_rejects_invalid_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a real PDF")

    with pytest.raises(ValueError, match="not a valid PDF"):
        load_pdf(path)


def test_load_pdf_wraps_page_extraction_errors(pdf_factory):
    path = pdf_factory("malformed.pdf")
    path.write_bytes(path.read_bytes().replace(b"endstream", b""))

    with pytest.raises(ValueError, match="PDF is malformed"):
        load_pdf(path)
