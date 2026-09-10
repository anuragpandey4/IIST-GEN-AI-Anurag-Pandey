from pathlib import Path

import pytest

from src.config import Settings
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


def _settings() -> Settings:
    return Settings(
        groq_api_key="test-key-not-used",
        llm_model="test-model",
        vector_db_path=".data/test-vector-db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        rag_relevance_threshold=1.5,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


@pytest.mark.asyncio
async def test_uninitialized_pipeline_rejects_chat(tmp_path: Path) -> None:
    pipeline = SupportPipeline(_settings(), tmp_path)
    with pytest.raises(ComponentNotReadyError):
        await pipeline.process("session-1", "hello")


@pytest.mark.asyncio
async def test_blank_session_id_rejected(tmp_path: Path) -> None:
    """Pipeline must reject blank session IDs."""
    pipeline = SupportPipeline(_settings(), tmp_path)
    pipeline.ready = True
    pipeline.workflow = True  # Fake as truthy to pass readiness check.
    with pytest.raises((AgentProcessingError, ValueError)):
        await pipeline.process("   ", "hello")


@pytest.mark.asyncio
async def test_blank_message_rejected(tmp_path: Path) -> None:
    """Pipeline must reject blank messages."""
    pipeline = SupportPipeline(_settings(), tmp_path)
    pipeline.ready = True
    pipeline.workflow = True
    with pytest.raises((AgentProcessingError, ValueError)):
        await pipeline.process("session-1", "   ")
