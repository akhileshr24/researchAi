"""FastAPI endpoints exposing the existing hybrid RAG pipeline.

The endpoints only orchestrate the RAG modules; retrieval, reranking and
answer generation reuse ``src.rag`` unchanged. Blocking model work runs in the
FastAPI threadpool so the async event loop stays responsive.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from src.api.rate_limit import limiter
from src.rag.ingestion import ingest_pdf
from src.rag.qa import ask_question
from src.rag.vectorstore import (
    delete_document,
    get_document,
    list_documents,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


# REQUEST AND RESPONSE SCHEMAS


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
    )

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty")

        return value


class SourceCitation(BaseModel):
    filename: str
    page: int
    chunk_index: int


class RagQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceCitation]


class RagUploadResponse(BaseModel):
    status: Literal["ingested", "replaced", "skipped"]
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    message: str | None = None


class RagDocumentsResponse(BaseModel):
    documents: list[dict]
    total: int


class RagDeleteResponse(BaseModel):
    document_id: str
    deleted: bool


# ENDPOINTS


@router.post("/upload", response_model=RagUploadResponse)
@limiter.limit("10/minute")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
) -> RagUploadResponse:
    """Validate, chunk, embed and store one uploaded PDF."""
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported",
        )

    content = await file.read()
    await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty",
        )

    # Keep the original filename so the stored document metadata matches
    # the uploaded file; Path(...).name strips any client-side directories.
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / Path(file.filename).name
            tmp_path.write_bytes(content)
            result = await run_in_threadpool(ingest_pdf, tmp_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("PDF ingestion failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to process the PDF. Please try again later.",
        ) from None

    return result


@router.post("/query", response_model=RagQueryResponse)
@limiter.limit("20/minute")
async def query_documents(
    request: Request,
    body: RagQueryRequest,
) -> RagQueryResponse:
    """Answer a question from the indexed documents."""
    if body.document_id is not None:
        document = await run_in_threadpool(get_document, body.document_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {body.document_id}",
            )

    try:
        result = await run_in_threadpool(
            ask_question,
            body.question,
            document_id=body.document_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("RAG query failed")

        raise HTTPException(
            status_code=500,
            detail="RAG query failed. Please try again later.",
        ) from None

    return {"question": body.question, **result}


@router.get("/documents", response_model=RagDocumentsResponse)
async def list_indexed_documents() -> RagDocumentsResponse:
    """List all indexed documents with their page and chunk counts."""
    try:
        documents = await run_in_threadpool(list_documents)
    except Exception:
        logger.exception("Listing indexed documents failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to list documents. Please try again later.",
        ) from None

    return {"documents": documents, "total": len(documents)}


@router.delete("/documents/{document_id}", response_model=RagDeleteResponse)
async def remove_indexed_document(document_id: str) -> RagDeleteResponse:
    """Delete one indexed document and all of its chunks."""
    deleted = await run_in_threadpool(delete_document, document_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}",
        )

    return {"document_id": document_id, "deleted": True}
