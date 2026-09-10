import tempfile
from pathlib import Path

import pytest

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.retriever import KnowledgeRetriever
from src.utils.errors import ComponentNotReadyError


def _test_settings(tmp_path: Path) -> Settings:
    """Create settings pointing to a temporary vector DB directory."""
    return Settings(
        groq_api_key="test-key-not-used",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vector_db"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        rag_relevance_threshold=1.5,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


def test_loader_preserves_source_names(knowledge_dir) -> None:
    documents = load_support_documents(knowledge_dir)
    assert {item.metadata["source"] for item in documents} == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(knowledge_dir) -> None:
    chunks = split_support_documents(load_support_documents(knowledge_dir))
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".md") for chunk in chunks)


@pytest.mark.asyncio
async def test_search_before_init_raises_error(knowledge_dir, tmp_path) -> None:
    """Searching before initialization must raise ComponentNotReadyError."""
    settings = _test_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    # Do NOT call initialize.
    with pytest.raises(ComponentNotReadyError):
        await retriever.search("shipping")


@pytest.mark.asyncio
async def test_blank_query_returns_empty(knowledge_dir, tmp_path) -> None:
    """A blank or whitespace query must return an empty list, not crash."""
    settings = _test_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    assert await retriever.search("") == []
    assert await retriever.search("   ") == []


@pytest.mark.asyncio
async def test_initialize_creates_store(knowledge_dir, tmp_path) -> None:
    """After initialization, the internal store must be set."""
    settings = _test_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    assert retriever._store is not None


@pytest.mark.asyncio
async def test_search_returns_correct_format(knowledge_dir, tmp_path) -> None:
    """Search results must be dicts with 'content' and 'source' keys."""
    settings = _test_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    results = await retriever.search("shipping delivery")
    assert isinstance(results, list)
    if results:
        for r in results:
            assert "content" in r
            assert "source" in r
            # Source must be a filename, not an absolute path.
            assert "/" not in r["source"] and "\\" not in r["source"]
            assert r["source"].endswith(".md")


@pytest.mark.asyncio
async def test_relevant_query_finds_shipping(knowledge_dir, tmp_path) -> None:
    """A shipping-related query must retrieve shipping.md content."""
    settings = _test_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    results = await retriever.search("How long does standard shipping take?")
    assert len(results) > 0
    sources = [r["source"] for r in results]
    assert "shipping.md" in sources


@pytest.mark.asyncio
async def test_initialization_is_idempotent(knowledge_dir, tmp_path) -> None:
    """Calling initialize() twice must not duplicate chunks or crash."""
    settings = _test_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()
    first_results = await retriever.search("returns refund policy")

    # Re-initialize — should be safe.
    await retriever.initialize()
    second_results = await retriever.search("returns refund policy")

    # Same number of results.
    assert len(first_results) == len(second_results)


@pytest.mark.asyncio
async def test_unknown_question_returns_few_or_no_results(knowledge_dir, tmp_path) -> None:
    """A question with no matching knowledge must return empty or very low results.

    This validates the relevance threshold filtering: queries unrelated to the
    knowledge base should not produce fabricated or irrelevant results.
    """
    settings = _test_settings(tmp_path)
    # Use a strict (low) distance threshold to demonstrate filtering.
    # L2 distance: lower threshold = only very close matches pass.
    settings_strict = Settings(
        groq_api_key=settings.groq_api_key,
        llm_model=settings.llm_model,
        vector_db_path=settings.vector_db_path,
        embedding_model=settings.embedding_model,
        rag_collection=settings.rag_collection + "-strict",
        rag_top_k=settings.rag_top_k,
        rag_relevance_threshold=0.5,  # Very strict: only near-exact matches.
        api_host=settings.api_host,
        api_port=settings.api_port,
        streamlit_host=settings.streamlit_host,
        streamlit_port=settings.streamlit_port,
    )
    retriever = KnowledgeRetriever(settings_strict, knowledge_dir)
    await retriever.initialize()

    # This question is completely unrelated to the knowledge base.
    results = await retriever.search(
        "What is the capital of France and who invented the airplane?"
    )
    # With a strict threshold, unrelated queries should return few/no results.
    assert len(results) <= 1
