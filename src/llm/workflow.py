from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.llm.prompts import (
    ANSWER_TEMPLATE,
    DECIDE_PROMPT,
    SYSTEM_PROMPT,
    TICKET_FOLLOWUP_PROMPT,
    TICKET_SUMMARY_PROMPT,
)
from src.models import TicketCreate
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository, create_ticket_tool
from src.utils.errors import AgentProcessingError


class SupportWorkflowState(TypedDict, total=False):
    """Typed state shared by the supplied LangGraph node skeletons."""

    session_id: str
    customer_message: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: list[dict[str, str]]
    route: str
    extracted_fields: dict[str, str]
    response_text: str
    sources: list[str]
    ticket_id: str | None


class IntentDecision(BaseModel):
    """Structured output schema for intent classification."""
    route: Literal["answer", "ticket"] = Field(
        description="'answer' for policy questions answerable from context, "
        "'ticket' for help requests, complaints, or ticket-related interactions"
    )
    extracted_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Customer details explicitly stated in this message. "
        "Keys: customer_name, customer_email, issue_description, category. "
        "Only include fields the customer explicitly provided."
    )


def build_support_workflow(
    model: BaseChatModel,
    retriever: KnowledgeRetriever,
    sessions: SessionStore,
    tickets: TicketRepository,
):
    """Build the agent graph with complete node logic.

    Dependencies (retriever, sessions, tickets) are injected via closure so each
    node has access to shared state without global variables.
    """

    async def retrieve(state: SupportWorkflowState) -> SupportWorkflowState:
        """Retrieve relevant knowledge chunks for the customer's message."""
        query = state.get("customer_message", "")
        chunks = await retriever.search(query)

        # Deduplicate by content while preserving source names.
        seen: set[str] = set()
        unique_chunks: list[dict[str, str]] = []
        for chunk in chunks:
            content = chunk["content"]
            if content not in seen:
                seen.add(content)
                unique_chunks.append(chunk)

        return {"retrieved_chunks": unique_chunks}

    async def decide(state: SupportWorkflowState) -> SupportWorkflowState:
        """Classify customer intent and extract any provided fields."""
        chunks = state.get("retrieved_chunks", [])
        message = state.get("customer_message", "")
        session_id = state.get("session_id", "")

        # Build context from retrieved chunks.
        context = "\n\n".join(
            f"[{c['source']}]: {c['content']}" for c in chunks
        ) if chunks else "No relevant knowledge context found."

        # Build conversation history for context.
        session = sessions.get_or_create(session_id)
        history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in session.history[-6:]  # Last 3 turns for context
        ) if session.history else "No previous conversation."

        prompt = DECIDE_PROMPT.format(
            context=context, history=history, message=message
        )

        try:
            structured_model = model.with_structured_output(IntentDecision)
            decision: IntentDecision = await structured_model.ainvoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            return {
                "route": decision.route,
                "extracted_fields": decision.extracted_fields,
            }
        except Exception:
            # On parse failure, default to answer route so we don't crash.
            return {"route": "answer", "extracted_fields": {}}

    async def answer(state: SupportWorkflowState) -> SupportWorkflowState:
        """Generate a grounded answer using only retrieved knowledge context."""
        chunks = state.get("retrieved_chunks", [])
        message = state.get("customer_message", "")
        session_id = state.get("session_id", "")

        # Build context from retrieved chunks.
        if chunks:
            context = "\n\n".join(
                f"[{c['source']}]: {c['content']}" for c in chunks
            )
            sources = list(dict.fromkeys(c["source"] for c in chunks))
        else:
            context = "No relevant information found in the knowledge base."
            sources = []

        # Build conversation history.
        session = sessions.get_or_create(session_id)
        history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in session.history[-6:]
        ) if session.history else "No previous conversation."

        prompt = ANSWER_TEMPLATE.format(
            context=context, history=history, message=message
        )

        response = await model.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )

        return {
            "response_text": response.content,
            "sources": sources,
            "ticket_id": state.get("ticket_id"),
        }

    async def collect_or_create(state: SupportWorkflowState) -> SupportWorkflowState:
        """Collect ticket fields over multiple turns, create when complete."""
        session_id = state.get("session_id", "")
        message = state.get("customer_message", "")
        session = sessions.get_or_create(session_id)

        # Idempotency: if ticket already exists for this session, return it.
        if session.ticket_id:
            return {
                "response_text": (
                    f"Your support ticket has already been created with ID "
                    f"**{session.ticket_id}**. Our team is working on it. "
                    f"Is there anything else I can help you with?"
                ),
                "sources": [],
                "ticket_id": session.ticket_id,
            }

        # Merge any newly extracted fields into the session.
        extracted = state.get("extracted_fields", {})
        if extracted.get("customer_name"):
            session.customer_name = extracted["customer_name"]
        if extracted.get("customer_email"):
            session.customer_email = extracted["customer_email"]
        if extracted.get("issue_description"):
            session.issue_description = extracted["issue_description"]
        if extracted.get("category"):
            # Validate category against allowed values.
            cat = extracted["category"].lower().strip()
            if cat in ("order", "payment", "account", "technical", "other"):
                session.category = cat

        # Check what fields are still missing.
        missing = session.missing_ticket_fields()

        if missing:
            # Ask for the next missing field.
            collected = {}
            if session.customer_name:
                collected["customer_name"] = session.customer_name
            if session.customer_email:
                collected["customer_email"] = session.customer_email
            if session.issue_description:
                collected["issue_description"] = session.issue_description
            if session.category:
                collected["category"] = session.category

            history = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in session.history[-6:]
            ) if session.history else "No previous conversation."

            prompt = TICKET_FOLLOWUP_PROMPT.format(
                collected_fields=collected or "None yet",
                missing_fields=", ".join(missing),
                history=history,
            )

            response = await model.ainvoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )

            return {
                "response_text": response.content,
                "sources": [],
                "ticket_id": None,
            }

        # All fields collected — generate summary and create ticket.
        try:
            # Generate a short summary using the LLM.
            summary_prompt = TICKET_SUMMARY_PROMPT.format(
                issue_description=session.issue_description
            )
            summary_response = await model.ainvoke(
                [HumanMessage(content=summary_prompt)]
            )
            summary = summary_response.content.strip()[:160]
            if len(summary) < 5:
                summary = (session.issue_description or "Support request")[:160]

            # Validate all fields via Pydantic before creating.
            ticket_data = TicketCreate(
                customer_name=session.customer_name,  # type: ignore[arg-type]
                customer_email=session.customer_email,  # type: ignore[arg-type]
                issue_description=session.issue_description,  # type: ignore[arg-type]
                category=session.category,  # type: ignore[arg-type]
                summary=summary,
            )

            # Create the ticket using the bound tool.
            ticket = tickets.create(session_id, ticket_data)
            session.ticket_id = ticket.ticket_id

            return {
                "response_text": (
                    f"I've created your support ticket successfully!\n\n"
                    f"**Ticket ID:** {ticket.ticket_id}\n"
                    f"**Summary:** {summary}\n"
                    f"**Category:** {session.category}\n\n"
                    f"Our support team will review your case and get back to you. "
                    f"Is there anything else I can help you with?"
                ),
                "sources": [],
                "ticket_id": ticket.ticket_id,
            }

        except Exception as exc:
            # Validation or creation failure — ask user to correct input.
            error_msg = str(exc)
            if "email" in error_msg.lower():
                session.customer_email = None
                return {
                    "response_text": (
                        "The email address provided doesn't appear to be valid. "
                        "Could you please provide a valid email address?"
                    ),
                    "sources": [],
                    "ticket_id": None,
                }
            if "category" in error_msg.lower():
                session.category = None
                return {
                    "response_text": (
                        "The category provided isn't valid. Please choose from: "
                        "order, payment, account, technical, or other."
                    ),
                    "sources": [],
                    "ticket_id": None,
                }
            # Generic validation error.
            return {
                "response_text": (
                    "I encountered an issue creating your ticket. "
                    "Let me re-collect the information. Could you please "
                    "provide your full name?"
                ),
                "sources": [],
                "ticket_id": None,
            }

    def select_route(state: SupportWorkflowState) -> str:
        """Select the next node based on the decide step's structured output."""
        route = state.get("route")
        if route in ("answer", "ticket"):
            return route
        raise AgentProcessingError(
            f"Unexpected route value: {route!r}. Expected 'answer' or 'ticket'."
        )

    graph = StateGraph(SupportWorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("answer", answer)
    graph.add_node("ticket", collect_or_create)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        select_route,
        {"answer": "answer", "ticket": "ticket"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("ticket", END)
    return graph.compile()
