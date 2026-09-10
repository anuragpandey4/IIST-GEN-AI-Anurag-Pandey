"""Tests for FastAPI endpoints including graceful error handling."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models import ChatResponse
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


@pytest.fixture
def client():
    """Create a test client with the pipeline mocked to avoid real initialization."""
    # Patch the pipeline to avoid real LLM/embedding initialization.
    with patch("src.api.server.pipeline") as mock_pipeline:
        mock_pipeline.ready = True
        mock_pipeline.tickets = type("MockRepo", (), {
            "get": lambda self, tid: None,
        })()

        from src.api.server import app
        yield TestClient(app, raise_server_exceptions=False)


def test_health_ready(client) -> None:
    """Health endpoint must return 200 when pipeline is ready."""
    with patch("src.api.server.pipeline") as mock_pipeline:
        mock_pipeline.ready = True
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


def test_health_not_ready(client) -> None:
    """Health endpoint must return 503 when pipeline is not ready."""
    with patch("src.api.server.pipeline") as mock_pipeline:
        mock_pipeline.ready = False
        response = client.get("/health")
        assert response.status_code == 503


def test_chat_invalid_input_422(client) -> None:
    """Chat endpoint must return 422 for invalid input (empty message)."""
    response = client.post("/chat", json={"session_id": "test", "message": ""})
    assert response.status_code == 422


def test_chat_missing_session_422(client) -> None:
    """Chat endpoint must return 422 for missing session_id."""
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 422


def test_ticket_not_found_404(client) -> None:
    """GET /tickets/{id} must return 404 for unknown ticket IDs."""
    with patch("src.api.server.pipeline") as mock_pipeline:
        mock_repo = type("MockRepo", (), {
            "get": lambda self, tid: None,
        })()
        mock_pipeline.tickets = mock_repo
        response = client.get("/tickets/CST-2026-9999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_graceful_model_failure_502(client) -> None:
    """When the LLM/pipeline raises AgentProcessingError, API must return 502.

    This tests the graded scenario: 'Graceful model/backend failure'.
    The API must NOT return a 500 with a stack trace.
    """
    with patch("src.api.server.pipeline") as mock_pipeline:
        mock_pipeline.ready = True
        mock_pipeline.process = AsyncMock(
            side_effect=AgentProcessingError("Model timeout during processing")
        )
        response = client.post(
            "/chat",
            json={"session_id": "test-failure", "message": "help me"},
        )
        assert response.status_code == 502
        data = response.json()
        assert "detail" in data
        # Must NOT contain stack trace information.
        assert "Traceback" not in data["detail"]


def test_graceful_component_unavailable_503(client) -> None:
    """When a component is not ready, API must return 503."""
    with patch("src.api.server.pipeline") as mock_pipeline:
        mock_pipeline.ready = True
        mock_pipeline.process = AsyncMock(
            side_effect=ComponentNotReadyError("Vector store not initialized")
        )
        response = client.post(
            "/chat",
            json={"session_id": "test-unavail", "message": "hello"},
        )
        assert response.status_code == 503
        assert "detail" in response.json()


def test_successful_chat_response(client) -> None:
    """A successful chat must return a well-formed ChatResponse."""
    with patch("src.api.server.pipeline") as mock_pipeline:
        mock_pipeline.ready = True
        mock_pipeline.process = AsyncMock(
            return_value=ChatResponse(
                success=True,
                session_id="test-ok",
                response="Standard shipping takes 3-5 business days.",
                sources=["shipping.md"],
                ticket_id=None,
            )
        )
        response = client.post(
            "/chat",
            json={"session_id": "test-ok", "message": "How long does shipping take?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["response"]
        assert "shipping.md" in data["sources"]
        assert data["ticket_id"] is None
