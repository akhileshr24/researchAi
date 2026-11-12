"""Tests for the RAG API endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.api import app

client = TestClient(app)

UPLOAD_RESULT = {
    "status": "ingested",
    "document_id": "abc123def456",
    "filename": "guide.pdf",
    "total_pages": 5,
    "total_chunks": 13,
}


def upload_pdf(name: str = "guide.pdf", content: bytes = b"%PDF-1.4 fake"):
    return client.post(
        "/rag/upload",
        files={"file": (name, content, "application/pdf")},
    )


# ---------- POST /rag/upload ----------


@patch("src.api.rag_routes.ingest_pdf")
def test_upload_pdf_success(mock_ingest):
    mock_ingest.return_value = UPLOAD_RESULT

    response = upload_pdf()

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ingested"
    assert data["document_id"] == "abc123def456"
    assert data["filename"] == "guide.pdf"
    assert data["total_pages"] == 5
    assert data["total_chunks"] == 13

    mock_ingest.assert_called_once()


@patch("src.api.rag_routes.ingest_pdf")
def test_upload_duplicate_returns_skipped_status(mock_ingest):
    mock_ingest.return_value = {
        **UPLOAD_RESULT,
        "status": "skipped",
        "message": "Document already ingested",
    }

    response = upload_pdf()

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert response.json()["message"] == "Document already ingested"


def test_upload_rejects_non_pdf_file():
    response = client.post(
        "/rag/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_empty_file():
    response = upload_pdf(content=b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


@patch("src.api.rag_routes.ingest_pdf")
def test_upload_rejects_invalid_pdf(mock_ingest):
    mock_ingest.side_effect = ValueError(
        "Could not read 'guide.pdf': the file is not a valid PDF"
    )

    response = upload_pdf()

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


@patch("src.api.rag_routes.ingest_pdf")
def test_upload_hides_internal_errors(mock_ingest):
    mock_ingest.side_effect = RuntimeError("chroma secret internals")

    response = upload_pdf()

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Failed to process the PDF. Please try again later."
    )


# ---------- POST /rag/query ----------


@patch("src.api.rag_routes.ask_question")
def test_rag_query_success(mock_ask):
    mock_ask.return_value = {
        "answer": "RAG combines retrieval with generation.",
        "sources": [
            {"filename": "guide.pdf", "page": 5, "chunk_index": 10}
        ],
    }

    response = client.post(
        "/rag/query",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is RAG?"
    assert data["answer"] == "RAG combines retrieval with generation."
    assert data["sources"] == [
        {"filename": "guide.pdf", "page": 5, "chunk_index": 10}
    ]

    mock_ask.assert_called_once_with(
        "What is RAG?",
        document_id=None,
    )


@patch("src.api.rag_routes.ask_question")
@patch("src.api.rag_routes.get_document")
def test_rag_query_with_document_id(mock_get_document, mock_ask):
    mock_get_document.return_value = {
        "document_id": "abc123def456",
        "filename": "guide.pdf",
        "total_pages": 5,
        "total_chunks": 13,
    }
    mock_ask.return_value = {"answer": "Answer.", "sources": []}

    response = client.post(
        "/rag/query",
        json={
            "question": "What is RAG?",
            "document_id": "abc123def456",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Answer."

    mock_get_document.assert_called_once_with("abc123def456")
    mock_ask.assert_called_once_with(
        "What is RAG?",
        document_id="abc123def456",
    )


@patch("src.api.rag_routes.get_document")
def test_rag_query_unknown_document_returns_404(mock_get_document):
    mock_get_document.return_value = None

    response = client.post(
        "/rag/query",
        json={
            "question": "What is RAG?",
            "document_id": "missing-id",
        },
    )

    assert response.status_code == 404
    assert "missing-id" in response.json()["detail"]


@patch("src.api.rag_routes.ask_question")
def test_rag_query_no_answer_has_empty_sources(mock_ask):
    mock_ask.return_value = {
        "answer": "I couldn't find relevant information in the documents.",
        "sources": [],
    }

    response = client.post(
        "/rag/query",
        json={"question": "What is the capital of France?"},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_rag_query_rejects_empty_question():
    response = client.post(
        "/rag/query",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_rag_query_rejects_missing_question():
    response = client.post("/rag/query", json={})

    assert response.status_code == 422


def test_rag_query_rejects_overlong_question():
    response = client.post(
        "/rag/query",
        json={"question": "A" * 1001},
    )

    assert response.status_code == 422


@patch("src.api.rag_routes.ask_question")
def test_rag_query_hides_internal_errors(mock_ask):
    mock_ask.side_effect = RuntimeError("LLM provider secret internals")

    response = client.post(
        "/rag/query",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "RAG query failed. Please try again later."
    )


# ---------- GET /rag/documents ----------


@patch("src.api.rag_routes.list_documents")
def test_list_documents_returns_summaries(mock_list):
    mock_list.return_value = [
        {
            "document_id": "doc-a",
            "filename": "a.pdf",
            "total_pages": 3,
            "total_chunks": 7,
        },
        {
            "document_id": "doc-b",
            "filename": "b.pdf",
            "total_pages": 1,
            "total_chunks": 2,
        },
    ]

    response = client.get("/rag/documents")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert data["documents"][0]["document_id"] == "doc-a"
    assert data["documents"][0]["total_pages"] == 3
    assert data["documents"][0]["total_chunks"] == 7


@patch("src.api.rag_routes.list_documents")
def test_list_documents_empty_store(mock_list):
    mock_list.return_value = []

    response = client.get("/rag/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": [], "total": 0}


@patch("src.api.rag_routes.list_documents")
def test_list_documents_hides_internal_errors(mock_list):
    mock_list.side_effect = RuntimeError("chroma secret internals")

    response = client.get("/rag/documents")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Failed to list documents. Please try again later."
    )


# ---------- DELETE /rag/documents/{document_id} ----------


@patch("src.api.rag_routes.delete_document")
def test_delete_document_success(mock_delete):
    mock_delete.return_value = True

    response = client.delete("/rag/documents/doc-a")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-a",
        "deleted": True,
    }

    mock_delete.assert_called_once_with("doc-a")


@patch("src.api.rag_routes.delete_document")
def test_delete_missing_document_returns_404(mock_delete):
    mock_delete.return_value = False

    response = client.delete("/rag/documents/missing-id")

    assert response.status_code == 404
    assert "missing-id" in response.json()["detail"]
