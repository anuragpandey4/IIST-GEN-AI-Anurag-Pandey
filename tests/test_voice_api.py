"""API-level tests for /voice/transcribe and /voice/synthesize endpoints.

Follows the same mocking pattern as test_api.py — patches module-level objects
in src.api.server to avoid real model loading.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with pipeline + voice mocked."""
    with (
        patch("src.api.server.pipeline") as mock_pipeline,
        patch("src.api.server.voice_pipeline") as mock_voice,
    ):
        mock_pipeline.ready = True
        mock_pipeline.tickets = type("MockRepo", (), {
            "get": lambda self, tid: None,
        })()

        mock_voice.transcribe = AsyncMock(
            return_value=("My payment was charged twice", 820)
        )
        mock_voice.synthesize = AsyncMock(
            return_value=(b"fake-audio-bytes", "audio/mpeg", 150)
        )

        from src.api.server import app

        yield TestClient(app, raise_server_exceptions=False), mock_voice


# ── Transcribe Endpoint ──────────────────────────────────────────────────────


def test_transcribe_success(client) -> None:
    tc, _ = client
    with patch("src.api.server.voice_ready", True):
        resp = tc.post(
            "/voice/transcribe",
            files={"file": ("test.wav", b"fake-audio", "audio/wav")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["transcript"] == "My payment was charged twice"
    assert data["processing_time_ms"] == 820


def test_transcribe_empty_file(client) -> None:
    tc, mock_voice = client
    mock_voice.transcribe = AsyncMock(
        side_effect=ValueError("Audio input is empty")
    )
    with patch("src.api.server.voice_ready", True):
        resp = tc.post(
            "/voice/transcribe",
            files={"file": ("test.wav", b"", "audio/wav")},
        )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_transcribe_stt_failure(client) -> None:
    tc, mock_voice = client
    mock_voice.transcribe = AsyncMock(side_effect=RuntimeError("STT crash"))
    with patch("src.api.server.voice_ready", True):
        resp = tc.post(
            "/voice/transcribe",
            files={"file": ("test.wav", b"audio-data", "audio/wav")},
        )
    assert resp.status_code == 502


def test_transcribe_voice_not_ready(client) -> None:
    tc, _ = client
    with patch("src.api.server.voice_ready", False):
        resp = tc.post(
            "/voice/transcribe",
            files={"file": ("test.wav", b"audio-data", "audio/wav")},
        )
    assert resp.status_code == 503
    assert "not ready" in resp.json()["detail"].lower()


# ── Synthesize Endpoint ──────────────────────────────────────────────────────


def test_synthesize_success(client) -> None:
    tc, _ = client
    with patch("src.api.server.voice_ready", True):
        resp = tc.post(
            "/voice/synthesize",
            json={"message_id": "msg-001", "text": "Shipping takes 3-5 days"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"fake-audio-bytes"


def test_synthesize_empty_text(client) -> None:
    tc, _ = client
    with patch("src.api.server.voice_ready", True):
        resp = tc.post(
            "/voice/synthesize",
            json={"message_id": "msg-001", "text": ""},
        )
    # Pydantic validation error (min_length=1)
    assert resp.status_code == 422


def test_synthesize_tts_failure(client) -> None:
    tc, mock_voice = client
    mock_voice.synthesize = AsyncMock(side_effect=RuntimeError("TTS crash"))
    with patch("src.api.server.voice_ready", True):
        resp = tc.post(
            "/voice/synthesize",
            json={"message_id": "msg-001", "text": "Generate audio for this"},
        )
    assert resp.status_code == 502


def test_synthesize_voice_not_ready(client) -> None:
    tc, _ = client
    with patch("src.api.server.voice_ready", False):
        resp = tc.post(
            "/voice/synthesize",
            json={"message_id": "msg-001", "text": "Hello"},
        )
    assert resp.status_code == 503
