"""Tests for answer generation and citations."""

from types import SimpleNamespace

import pytest

from src.rag import qa


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="Grounded answer [a.pdf, page 1]")


def _scored(chunk, score=5.0):
    return {**chunk, "rerank_score": score}


def test_ask_question_builds_context_and_citations(
    monkeypatch,
    chunk_factory,
):
    fake_llm = FakeLLM()
    candidates = [chunk_factory(text="candidate")]
    results = [
        chunk_factory(
            text="First relevant fact.",
            page=1,
            chunk_index=0,
        ),
        chunk_factory(
            text="Second relevant fact.",
            page=1,
            chunk_index=1,
        ),
        chunk_factory(
            text="Fact from another document.",
            document_id="doc-b",
            filename="b.pdf",
            page=2,
            chunk_index=0,
        ),
    ]

    monkeypatch.setattr(
        qa,
        "search_hybrid",
        lambda question, top_k: candidates,
    )
    monkeypatch.setattr(
        qa,
        "rerank",
        lambda question, chunks, top_k: [
            _scored(chunk) for chunk in results
        ],
    )
    monkeypatch.setattr(qa, "get_llm", lambda provider: fake_llm)

    answer = qa.ask_question("What are the facts?")

    assert answer["answer"] == "Grounded answer [a.pdf, page 1]"
    assert answer["sources"] == [
        {"filename": "a.pdf", "page": 1},
        {"filename": "b.pdf", "page": 2},
    ]
    assert "First relevant fact." in fake_llm.prompts[0]
    assert "a.pdf, page 1" in fake_llm.prompts[0]
    assert "What are the facts?" in fake_llm.prompts[0]


def test_ask_question_handles_empty_retrieval(monkeypatch):
    monkeypatch.setattr(
        qa,
        "search_hybrid",
        lambda question, top_k: [],
    )
    monkeypatch.setattr(
        qa,
        "rerank",
        lambda question, chunks, top_k: [],
    )
    monkeypatch.setattr(
        qa,
        "get_llm",
        lambda provider: (_ for _ in ()).throw(
            AssertionError("LLM must not be called")
        ),
    )

    assert qa.ask_question("Unknown question") == {
        "answer": qa.NO_ANSWER,
        "sources": [],
    }


def test_ask_question_rejects_empty_question():
    with pytest.raises(ValueError, match="cannot be empty"):
        qa.ask_question("   ")


def test_ask_question_filters_irrelevant_results(monkeypatch, chunk_factory):
    fake_llm = FakeLLM()
    relevant = _scored(
        chunk_factory(text="Relevant fact.", page=1, chunk_index=0),
        5.0,
    )
    irrelevant = _scored(
        chunk_factory(
            text="Irrelevant fact.",
            document_id="doc-b",
            filename="b.pdf",
            page=2,
            chunk_index=0,
        ),
        -11.0,
    )

    monkeypatch.setattr(
        qa,
        "search_hybrid",
        lambda question, top_k: [relevant, irrelevant],
    )
    monkeypatch.setattr(
        qa,
        "rerank",
        lambda question, chunks, top_k: [relevant, irrelevant],
    )
    monkeypatch.setattr(qa, "get_llm", lambda provider: fake_llm)

    answer = qa.ask_question("What is a relevant fact?")

    assert answer["sources"] == [{"filename": "a.pdf", "page": 1}]
    assert "Relevant fact." in fake_llm.prompts[0]
    assert "Irrelevant fact." not in fake_llm.prompts[0]


def test_ask_question_returns_no_answer_when_all_results_are_irrelevant(
    monkeypatch,
    chunk_factory,
):
    irrelevant = _scored(
        chunk_factory(text="Irrelevant fact.", chunk_index=0),
        -11.0,
    )

    monkeypatch.setattr(
        qa,
        "search_hybrid",
        lambda question, top_k: [irrelevant],
    )
    monkeypatch.setattr(
        qa,
        "rerank",
        lambda question, chunks, top_k: [irrelevant],
    )
    monkeypatch.setattr(
        qa,
        "get_llm",
        lambda provider: (_ for _ in ()).throw(
            AssertionError("LLM must not be called")
        ),
    )

    assert qa.ask_question("What is the capital of France?") == {
        "answer": qa.NO_ANSWER,
        "sources": [],
    }
