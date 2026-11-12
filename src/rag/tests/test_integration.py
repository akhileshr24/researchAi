"""End-to-end test with real embeddings and a real cross-encoder.

The LLM call is replaced by a fake so the test does not need an API key.
"""

from types import SimpleNamespace

import chromadb

from src.config.settings import MIN_RERANK_SCORE
from src.rag import qa, vectorstore
from src.rag.hybrid_retriever import search_hybrid
from src.rag.ingestion import ingest_pdf
from src.rag.reranker import rerank
from src.rag.vectorstore import list_documents


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="Grounded answer [ai.pdf, page 2]")


def test_full_pipeline_with_two_pdfs(tmp_path, pdf_factory, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        name="integration_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(
        vectorstore,
        "_get_collection",
        lambda: collection,
    )

    ai_pdf = pdf_factory("ai.pdf", [
        ["Artificial intelligence systems learn patterns from data."],
        [
            "Hybrid retrieval combines vector search with BM25 keyword "
            "search and fuses both ranked lists using reciprocal rank fusion."
        ],
    ])
    cooking_pdf = pdf_factory("cooking.pdf", [
        ["Sourdough bread needs flour, water, salt and a starter."],
    ])

    assert ingest_pdf(ai_pdf)["status"] == "ingested"
    assert ingest_pdf(cooking_pdf)["status"] == "ingested"
    assert {document["filename"] for document in list_documents()} == {
        "ai.pdf",
        "cooking.pdf",
    }

    question = (
        "How does hybrid retrieval combine vector search and BM25?"
    )
    candidates = search_hybrid(question, top_k=5)

    assert candidates
    assert candidates[0]["filename"] == "ai.pdf"
    assert candidates[0]["page"] == 2

    reranked = rerank(question, candidates, top_k=2)

    assert reranked
    assert reranked[0]["filename"] == "ai.pdf"
    assert reranked[0]["page"] == 2
    assert "rerank_score" in reranked[0]
    assert reranked[0]["rerank_score"] >= MIN_RERANK_SCORE

    unrelated_question = "What is the capital of France?"
    unrelated_candidates = search_hybrid(unrelated_question, top_k=5)
    unrelated_ranked = rerank(
        unrelated_question,
        unrelated_candidates,
        top_k=5,
    )

    assert unrelated_ranked
    assert all(
        result["rerank_score"] < MIN_RERANK_SCORE
        for result in unrelated_ranked
    )

    fake_llm = FakeLLM()
    monkeypatch.setattr(qa, "get_llm", lambda provider: fake_llm)

    answer = qa.ask_question(question)

    assert answer["answer"] == "Grounded answer [ai.pdf, page 2]"
    assert any(
        source["filename"] == "ai.pdf" and source["page"] == 2
        for source in answer["sources"]
    )
    assert "reciprocal rank fusion" in fake_llm.prompts[0]
