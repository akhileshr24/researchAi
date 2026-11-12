"""Tests for PDF ingestion and document lifecycle."""

import pytest

from src.rag import ingestion
from src.rag.vectorstore import get_document, list_documents, search


def test_ingest_pdf_stores_document(pdf_factory, temp_store):
    path = pdf_factory("guide.pdf", [
        ["Vector search retrieves semantically similar chunks."],
        ["BM25 matches keyword overlap."],
    ])

    result = ingestion.ingest_pdf(path)

    assert result["status"] == "ingested"
    assert result["filename"] == "guide.pdf"
    assert result["total_pages"] == 2
    assert result["total_chunks"] >= 2
    assert len(list_documents()) == 1


def test_ingest_reports_total_pages_with_trailing_blank_page(
    pdf_factory,
    temp_store,
):
    path = pdf_factory("guide.pdf", [["Only text page."], []])

    result = ingestion.ingest_pdf(path)

    assert result["total_pages"] == 2
    assert list_documents()[0]["total_pages"] == 2
    assert get_document(result["document_id"])["total_pages"] == 2


def test_ingest_pdf_prevents_duplicates(pdf_factory, temp_store):
    path = pdf_factory("guide.pdf")

    ingestion.ingest_pdf(path)
    result = ingestion.ingest_pdf(path)

    assert result["status"] == "skipped"
    assert result["message"] == "Document already ingested"
    assert len(list_documents()) == 1


def test_duplicate_ingestion_does_not_reembed(
    pdf_factory,
    temp_store,
    monkeypatch,
):
    path = pdf_factory("guide.pdf")
    ingestion.ingest_pdf(path)
    monkeypatch.setattr(
        ingestion,
        "add_chunks",
        lambda chunks: (_ for _ in ()).throw(
            AssertionError("duplicate must not be embedded again")
        ),
    )

    assert ingestion.ingest_pdf(path)["status"] == "skipped"


def test_ingest_pdf_replace_rebuilds_document(pdf_factory, temp_store):
    path = pdf_factory("guide.pdf", [["First version content."]])
    ingestion.ingest_pdf(path)

    result = ingestion.ingest_pdf(path, replace=True)

    assert result["status"] == "replaced"
    assert len(list_documents()) == 1


def test_replace_removes_old_version_with_same_filename(
    pdf_factory,
    temp_store,
):
    path = pdf_factory("guide.pdf", [["vector search embeddings"]])
    first = ingestion.ingest_pdf(path)

    path = pdf_factory("guide.pdf", [["sourdough bread starter"]])
    second = ingestion.ingest_pdf(path, replace=True)

    assert first["document_id"] != second["document_id"]
    documents = list_documents()
    assert len(documents) == 1
    assert documents[0]["document_id"] == second["document_id"]

    results = search("sourdough bread")
    assert results
    assert all("sourdough" in result["text"] for result in results)


def test_ingest_multiple_pdfs_searches_all_documents(
    pdf_factory,
    temp_store,
):
    ingestion.ingest_pdf(pdf_factory("ai.pdf", [["vector search embeddings"]]))
    ingestion.ingest_pdf(pdf_factory("cooking.pdf", [["sourdough bread starter"]]))

    assert len(list_documents()) == 2

    results = search("sourdough bread")

    assert results
    assert results[0]["filename"] == "cooking.pdf"


def test_ingest_missing_file(temp_store, tmp_path):
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        ingestion.ingest_pdf(tmp_path / "missing.pdf")


def test_ingest_rejects_non_pdf(tmp_path, temp_store):
    path = tmp_path / "notes.txt"
    path.write_text("plain text", encoding="utf-8")

    with pytest.raises(ValueError, match="Not a PDF file"):
        ingestion.ingest_pdf(path)


def test_ingest_rejects_invalid_pdf(tmp_path, temp_store):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a real PDF")

    with pytest.raises(ValueError, match="not a valid PDF"):
        ingestion.ingest_pdf(path)


def test_ingest_rejects_pdf_without_text(pdf_factory, temp_store):
    path = pdf_factory("blank.pdf", [[]])

    with pytest.raises(ValueError, match="No extractable text"):
        ingestion.ingest_pdf(path)

    assert list_documents() == []
