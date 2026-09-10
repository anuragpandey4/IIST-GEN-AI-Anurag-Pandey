from src.sessions.store import ConversationState, SessionStore


def test_session_reports_only_missing_ticket_fields() -> None:
    # This supplied test establishes the expected collection order. Add tests for
    # an empty state, a complete state, and updates made across multiple turns.
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")

    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_empty_state_reports_all_fields_missing() -> None:
    """A brand new session should report all four ticket fields as missing."""
    state = ConversationState()
    assert state.missing_ticket_fields() == [
        "customer_name",
        "customer_email",
        "issue_description",
        "category",
    ]


def test_complete_state_reports_no_fields_missing() -> None:
    """A fully populated session should have no missing fields."""
    state = ConversationState(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="My order is delayed",
        category="order",
    )
    assert state.missing_ticket_fields() == []


def test_sessions_are_isolated_across_ids() -> None:
    """Different session IDs must not share state."""
    store = SessionStore()
    session_a = store.get_or_create("session-a")
    session_a.customer_name = "Alice"

    session_b = store.get_or_create("session-b")
    assert session_b.customer_name is None
    assert session_a.customer_name == "Alice"


def test_same_session_id_returns_same_state() -> None:
    """The same session ID must return the same ConversationState object."""
    store = SessionStore()
    first = store.get_or_create("session-1")
    first.customer_name = "Alice"

    second = store.get_or_create("session-1")
    assert second.customer_name == "Alice"
    assert first is second


def test_history_preserves_order() -> None:
    """Conversation history must maintain insertion order."""
    state = ConversationState()
    state.history.append({"role": "user", "content": "Hello"})
    state.history.append({"role": "assistant", "content": "Hi there!"})
    state.history.append({"role": "user", "content": "Help me"})

    assert len(state.history) == 3
    assert state.history[0]["role"] == "user"
    assert state.history[1]["role"] == "assistant"
    assert state.history[2]["content"] == "Help me"


def test_partial_update_keeps_existing_fields() -> None:
    """Updating one field must not reset previously collected fields."""
    state = ConversationState(customer_name="Alice")
    state.customer_email = "alice@example.com"

    assert state.customer_name == "Alice"
    assert state.customer_email == "alice@example.com"
    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_blank_session_id_raises_error() -> None:
    """A blank session ID must be rejected."""
    store = SessionStore()
    import pytest

    with pytest.raises(ValueError, match="must not be blank"):
        store.get_or_create("   ")


def test_completed_session_retains_ticket_id() -> None:
    """Once a ticket ID is stored, it persists across accesses."""
    store = SessionStore()
    session = store.get_or_create("session-ticket")
    session.ticket_id = "CST-2026-0001"

    same_session = store.get_or_create("session-ticket")
    assert same_session.ticket_id == "CST-2026-0001"
