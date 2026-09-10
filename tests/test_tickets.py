import pytest
from pydantic import ValidationError

from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    # The repository is provided working code. This test is also an integration
    # contract for the eventual agent tool: one session must produce one ticket.
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )

    first = repository.create("session-1", request)
    second = repository.create("session-1", request)

    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_unique_ids_across_sessions() -> None:
    """Different sessions must receive different ticket IDs."""
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )

    ticket_a = repository.create("session-a", request)
    ticket_b = repository.create("session-b", request)

    assert ticket_a.ticket_id != ticket_b.ticket_id
    assert len(list(repository.all())) == 2


def test_field_preservation() -> None:
    """Created ticket must preserve all validated fields exactly."""
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Jane Doe",
        customer_email="jane@example.com",
        issue_description="Cannot access my account after password reset",
        category="account",
        summary="Account access issue post password reset",
    )

    ticket = repository.create("session-fields", request)
    assert ticket.customer_name == "Jane Doe"
    assert ticket.customer_email == "jane@example.com"
    assert ticket.category == "account"
    assert ticket.issue_description == "Cannot access my account after password reset"
    assert ticket.status == "open"


def test_get_existing_ticket() -> None:
    """Repository.get must return the ticket for a known ID."""
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Issue description here",
        category="technical",
        summary="Technical support request",
    )
    ticket = repository.create("session-get", request)
    found = repository.get(ticket.ticket_id)
    assert found is not None
    assert found.ticket_id == ticket.ticket_id


def test_get_unknown_ticket_returns_none() -> None:
    """Repository.get must return None for an unknown ticket ID."""
    repository = TicketRepository()
    assert repository.get("CST-2026-9999") is None


def test_category_enum_validation() -> None:
    """TicketCreate must reject invalid category values via Pydantic."""
    with pytest.raises(ValidationError):
        TicketCreate(
            customer_name="Test",
            customer_email="test@example.com",
            issue_description="Some issue description",
            category="billing",  # Not in Literal enum
            summary="Invalid category test",
        )


def test_email_validation() -> None:
    """TicketCreate must reject malformed email addresses via Pydantic."""
    with pytest.raises(ValidationError):
        TicketCreate(
            customer_name="Test",
            customer_email="not-an-email",
            issue_description="Some issue description",
            category="order",
            summary="Bad email test",
        )


def test_short_description_rejected() -> None:
    """TicketCreate must reject descriptions shorter than 5 characters."""
    with pytest.raises(ValidationError):
        TicketCreate(
            customer_name="Test",
            customer_email="test@example.com",
            issue_description="Hi",  # Too short
            category="order",
            summary="Short desc test",
        )


def test_idempotency_returns_first_ticket_on_retry() -> None:
    """Repeated create calls for the same session must return the original ticket."""
    repository = TicketRepository()
    request1 = TicketCreate(
        customer_name="User One",
        customer_email="one@example.com",
        issue_description="First issue description",
        category="order",
        summary="First request",
    )
    request2 = TicketCreate(
        customer_name="User Two",
        customer_email="two@example.com",
        issue_description="Different issue entirely",
        category="payment",
        summary="Second request",
    )

    first = repository.create("session-idem", request1)
    second = repository.create("session-idem", request2)

    assert first.ticket_id == second.ticket_id
    # The original ticket data is preserved, not overwritten.
    assert repository.get(first.ticket_id).customer_name == "User One"
