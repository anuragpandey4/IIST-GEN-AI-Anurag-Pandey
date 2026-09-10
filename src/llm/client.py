from langchain_groq import ChatGroq

from src.config import Settings


def build_chat_model(settings: Settings) -> ChatGroq:
    """Create the configured LangChain chat-model adapter using Groq.

    Groq serves open-source models (Llama, Mixtral) via an OpenAI-compatible
    chat-completions API with very low latency. The free tier provides generous
    rate limits suitable for development and evaluation.
    """
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0.2,
        max_retries=1,
        timeout=30,
        max_tokens=512,
    )
