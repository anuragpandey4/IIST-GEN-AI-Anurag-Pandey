from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Environment-controlled model and retrieval settings.

    Add new provider or embedding options here when required. Keep safe defaults
    suitable for local development and document every new variable in both
    ``.env.example`` and the project README.
    """
    groq_api_key: str
    llm_model: str
    vector_db_path: str
    embedding_model: str
    rag_collection: str
    rag_top_k: int
    rag_relevance_threshold: float
    api_host: str
    api_port: int
    streamlit_host: str
    streamlit_port: int


def load_settings() -> Settings:
    """Load the small, provider-neutral configuration used by the starter."""
    load_dotenv()
    # Configuration rules:
    # - Never commit a real API key.
    # - Avoid absolute developer-machine paths.
    # - Keep model identity configurable so reviewers can use another compatible
    #   open-source endpoint without editing source code.
    # - Validate required configuration during startup, not on the first request.
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key or groq_api_key.startswith("gsk_your"):
        raise ValueError(
            "GROQ_API_KEY is missing or still set to the placeholder. "
            "Get a free key at https://console.groq.com and add it to .env"
        )
    return Settings(
        groq_api_key=groq_api_key,
        llm_model=os.getenv("LLM_MODEL", "qwen/qwen3.8-27b"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", ".data/vector_db"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        rag_collection=os.getenv("RAG_COLLECTION", "customer-support"),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        rag_relevance_threshold=float(
            os.getenv("RAG_RELEVANCE_THRESHOLD", "1.5")
        ),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        streamlit_host=os.getenv("STREAMLIT_HOST", "127.0.0.1"),
        streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
    )
