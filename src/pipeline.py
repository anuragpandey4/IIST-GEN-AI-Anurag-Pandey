from __future__ import annotations

import logging
from pathlib import Path

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError

logger = logging.getLogger(__name__)


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared components once during FastAPI startup."""
        # The order is intentional: retrieval must be ready before a workflow
        # capable of accepting traffic is exposed. If either step fails, leave
        # ``ready`` false and let the FastAPI lifespan fail clearly.
        await self.retriever.initialize()
        self.workflow = build_support_workflow(
            self.model, self.retriever, self.sessions, self.tickets
        )
        self.ready = True

    async def process(self, session_id: str, message: str) -> ChatResponse:
        """Bind one complete customer turn through the agent workflow.

        Validates input, maintains session state, invokes the LangGraph workflow,
        and returns a structured response with sources and optional ticket ID.
        """
        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError("Support pipeline is not ready")

        # Validate and normalize input.
        session_id = session_id.strip()
        message = message.strip()
        if not session_id:
            raise AgentProcessingError("Session ID must not be blank")
        if not message:
            raise AgentProcessingError("Message must not be blank")

        # Obtain isolated session state.
        session = self.sessions.get_or_create(session_id)

        # Append the customer turn to conversation history.
        session.history.append({"role": "user", "content": message})

        try:
            # Build the initial workflow state.
            initial_state = {
                "session_id": session_id,
                "customer_message": message,
                "messages": [],
                "retrieved_chunks": [],
                "route": "",
                "extracted_fields": {},
                "response_text": "",
                "sources": [],
                "ticket_id": session.ticket_id,
            }

            # Invoke the LangGraph workflow.
            result = await self.workflow.ainvoke(initial_state)

            # Extract and validate the response.
            response_text = result.get("response_text", "").strip()
            if not response_text:
                raise AgentProcessingError("Workflow produced an empty response")

            sources = result.get("sources", [])
            ticket_id = result.get("ticket_id")

            # Append the successful assistant turn to history.
            session.history.append({"role": "assistant", "content": response_text})

            return ChatResponse(
                success=True,
                session_id=session_id,
                response=response_text,
                sources=sources,
                ticket_id=ticket_id,
            )

        except (ComponentNotReadyError, AgentProcessingError):
            # Re-raise known errors without wrapping.
            raise
        except Exception as exc:
            # Wrap unknown errors to prevent exposing internals.
            logger.exception("Unexpected error in pipeline.process: %s", exc)
            raise AgentProcessingError(
                "An error occurred while processing your request. "
                "Please try again."
            ) from exc
