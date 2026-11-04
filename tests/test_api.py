from fastapi.testclient import TestClient

from src.api.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ResearchAI API is running"
    }

from unittest.mock import MagicMock, patch


@patch("src.api.api.get_llm")
def test_chat_success(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.bind.return_value.invoke.return_value.content = (
        "Hello! I am ResearchAI."
    )
    mock_get_llm.return_value = mock_llm

    response = client.post(
        "/chat",
        json={
            "message": "Hello"
        }
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Hello"
    assert response.json()["response"] == "Hello! I am ResearchAI."

    mock_get_llm.assert_called_once()

def test_chat_empty_message():
    response = client.post(
        "/chat",
        json={"message": ""}
    )

    assert response.status_code == 422


def test_chat_message_too_long():
    response = client.post(
        "/chat",
        json={"message": "A" * 1001}
    )

    assert response.status_code == 422


@patch("src.api.api.get_llm")
def test_chat_provider_error(mock_get_llm):
    mock_get_llm.side_effect = RuntimeError("Provider unavailable")

    response = client.post(
        "/chat",
        json={"message": "Hello"}
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Chat failed. Please try again later."
    }


@patch("src.api.api.research_graph.invoke")
def test_research_success(mock_research):
    mock_research.return_value = {
        "answer": "Python is a popular programming language.",
        "content": [
            {
                "title": "Python Documentation",
                "url": "https://www.python.org/",
                "content": "Python is a programming language."
            }
        ]
    }

    response = client.post(
        "/research",
        json={"query": "Python programming"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "Python programming"
    assert data["report"] == "Python is a popular programming language."
    assert len(data["sources"]) == 1

    mock_research.assert_called_once()   

def test_research_query_too_short():
    response = client.post(
        "/research",
        json={"query": "AI"}
    )

    assert response.status_code == 422


def test_research_query_too_long():
    response = client.post(
        "/research",
        json={"query": "A" * 501}
    )

    assert response.status_code == 422     
@patch("src.api.api.research_graph.invoke")
def test_research_workflow_error(mock_research):
    mock_research.side_effect = RuntimeError("Search provider unavailable")

    response = client.post(
        "/research",
        json={"query": "Python programming"}
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Research failed. Please try again later."
    }
@patch("src.api.api.get_llm")
def test_chat_rate_limit(mock_get_llm):
    from fastapi.testclient import TestClient

    mock_llm = MagicMock()
    mock_llm.bind.return_value.invoke.return_value.content = "Test response"
    mock_get_llm.return_value = mock_llm

    # Use a separate IP so earlier tests don't affect this test.
    rate_client = TestClient(
        app,
        client=("192.0.2.100", 12345)
    )

    responses = []

    for _ in range(11):
        response = rate_client.post(
            "/chat",
            json={"message": "Hello"}
        )
        responses.append(response.status_code)

    assert responses.count(200) == 10
    assert responses.count(429) == 1


@patch("src.api.api.research_graph.invoke")
def test_research_rate_limit(mock_research):
    mock_research.return_value = {
        "answer": "Test research report",
        "content": []
    }

    rate_client = TestClient(
        app,
        client=("192.0.2.101", 12345)
    )

    responses = []

    for _ in range(4):
        response = rate_client.post(
            "/research",
            json={"query": "Python programming"}
        )
        responses.append(response.status_code)

    assert responses.count(200) == 3
    assert responses.count(429) == 1    

def test_chat_missing_message():
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_research_missing_query():
    response = client.post("/research", json={})
    assert response.status_code == 422

def test_chat_whitespace_message():
    response = client.post(
        "/chat",
        json={"message": "   "}
    )

    assert response.status_code == 422


def test_research_whitespace_query():
    response = client.post(
        "/research",
        json={"query": "   "}
    )

    assert response.status_code == 422


def test_chat_invalid_history_role():
    response = client.post(
        "/chat",
        json={
            "message": "Hello",
            "history": [
                {
                    "role": "system",
                    "content": "Invalid role"
                }
            ]
        }
    )

    assert response.status_code == 422


def test_chat_history_exceeds_limit():
    response = client.post(
        "/chat",
        json={
            "message": "Hello",
            "history": [
                {"role": "user", "content": f"Message {i}"}
                for i in range(7)
            ]
        }
    )

    assert response.status_code == 422
