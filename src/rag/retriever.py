from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma retrieval scaffold with stable output contracts."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    async def initialize(self) -> None:
        """Initialize the Chroma vector store with knowledge-base documents.

        Uses stable document IDs derived from source filename and content to
        prevent duplicate chunks across API restarts. The embedding model is
        downloaded once and cached by HuggingFace Hub.
        """
        documents = split_support_documents(load_support_documents(self.documents_dir))
        embeddings = build_embeddings(self.settings)

        # Resolve and create the persistence directory.
        persist_dir = Path(self.settings.vector_db_path)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Build the Chroma store with the configured collection and embeddings.
        store = Chroma(
            collection_name=self.settings.rag_collection,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

        # Generate stable IDs from source + content so restarts don't duplicate.
        stable_ids = [
            hashlib.md5(
                (doc.metadata.get("source", "") + doc.page_content).encode()
            ).hexdigest()
            for doc in documents
        ]

        # Ensure only safe filenames are stored (no absolute paths).
        for doc in documents:
            doc.metadata["source"] = Path(doc.metadata.get("source", "unknown")).name

        # Upsert: Chroma deduplicates by ID automatically.
        store.add_documents(documents, ids=stable_ids)

        self._store = store

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        """Retrieve relevant knowledge chunks for a customer query.

        Returns a list of plain dictionaries with ``content`` and ``source``
        keys, sorted by relevance (highest first). Results below the
        configured relevance threshold are excluded to prevent grounding
        the model on irrelevant content.

        Args:
            query: The customer's question or message.
            limit: Maximum number of results. Falls back to ``settings.rag_top_k``.

        Returns:
            List of ``{"content": str, "source": str}`` dicts, possibly empty.

        Raises:
            ComponentNotReadyError: If ``initialize()`` has not completed.
        """
        if self._store is None:
            raise ComponentNotReadyError(
                "KnowledgeRetriever has not been initialized"
            )

        # Reject blank or whitespace-only queries gracefully.
        if not query or not query.strip():
            return []

        k = limit or self.settings.rag_top_k
        threshold = self.settings.rag_relevance_threshold

        # Chroma returns L2 distances via similarity_search_with_score:
        # lower distance = more similar. We use the threshold as a maximum
        # L2 distance cutoff (default 1.5). Results above this distance are
        # considered irrelevant and excluded.
        results_with_scores = self._store.similarity_search_with_score(
            query, k=k
        )

        # Filter by distance threshold and normalize to plain dicts.
        # Sort by distance ascending (most relevant first).
        results_with_scores.sort(key=lambda x: x[1])

        chunks: list[dict[str, str]] = []
        seen_content: set[str] = set()
        for doc, distance in results_with_scores:
            if distance > threshold:
                continue
            content = doc.page_content.strip()
            if content in seen_content:
                continue
            seen_content.add(content)
            # Only expose safe filenames, never absolute paths.
            source = Path(doc.metadata.get("source", "unknown")).name
            chunks.append({"content": content, "source": source})

        return chunks
