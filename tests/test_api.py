"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI REST API endpoints.
The RAG chain is mocked so these tests are fast and offline.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_chain():
    """A mock RAG chain that returns a canned answer."""
    chain = MagicMock()
    chain.invoke = MagicMock(return_value="Rohanta is an AI/ML Engineer.")
    return chain


@pytest.fixture
def client(mock_chain):
    """Create a TestClient with the RAG chain patched."""
    with patch("src.api.main.get_chain", return_value=mock_chain):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_shape(client):
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "index_dir" in data


# ── /chat ─────────────────────────────────────────────────────────────────────

def test_chat_returns_200(client):
    response = client.post("/chat", json={"question": "Who is Rohanta?"})
    assert response.status_code == 200


def test_chat_response_contains_answer(client):
    response = client.post("/chat", json={"question": "Who is Rohanta?"})
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_chat_response_echoes_question(client):
    question = "Tell me about Rohanta's skills."
    response = client.post("/chat", json={"question": question})
    data = response.json()
    assert data["question"] == question


def test_chat_empty_question_rejected(client):
    """Empty question string should return 422 validation error."""
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_missing_body_rejected(client):
    """Missing request body should return 422."""
    response = client.post("/chat", json={})
    assert response.status_code == 422
