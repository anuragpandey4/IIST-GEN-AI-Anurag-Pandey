import os
import uuid

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Support", page_icon="🎧")
st.title("🎧 Customer Support")

# ─── Session defaults ────────────────────────────────────────────────────────
st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])
st.session_state.setdefault("audio_cache", {})
st.session_state.setdefault("voice_transcript", "")
st.session_state.setdefault("transcript_sent", False)
st.session_state.setdefault("needs_transcription", False)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.audio_cache = {}
        st.session_state.voice_transcript = ""
        st.session_state.transcript_sent = False
        st.session_state.needs_transcription = False
        st.rerun()
    st.caption(f"Session: `{st.session_state.session_id}`")


# ─── Callback: fires ONCE when audio_input changes ──────────────────────────
def on_audio_change():
    """Called exactly once when the user records new audio."""
    st.session_state.needs_transcription = True
    st.session_state.transcript_sent = False
    st.session_state.voice_transcript = ""


# ─── Render chat history ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                st.caption("📎 Sources: " + ", ".join(msg["sources"]))
            if msg.get("ticket_id"):
                st.success(f"✅ Ticket created: **{msg['ticket_id']}**")
            mid = msg.get("message_id", "")
            if mid:
                # Show cached audio player.
                if mid in st.session_state.audio_cache:
                    st.audio(st.session_state.audio_cache[mid], format="audio/mpeg")
                # Speaker button.
                label = "🔊 Play again" if mid in st.session_state.audio_cache else "🔊 Play response"
                if st.button(label, key=f"speak-{mid}"):
                    if mid not in st.session_state.audio_cache:
                        try:
                            r = httpx.post(
                                f"{API_BASE_URL}/voice/synthesize",
                                json={"message_id": mid, "text": msg["content"]},
                                timeout=60,
                            )
                            if r.status_code == 200:
                                st.session_state.audio_cache[mid] = r.content
                        except Exception:
                            pass
                    st.rerun()


# ─── Process pending message ─────────────────────────────────────────────────
if "pending_message" in st.session_state:
    text = st.session_state.pop("pending_message")
    st.session_state.messages.append({"role": "user", "content": text})
    try:
        r = httpx.post(
            f"{API_BASE_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": text},
            timeout=httpx.Timeout(connect=30, read=60, write=30, pool=30),
        )
        if r.status_code == 200:
            d = r.json()
            st.session_state.messages.append({
                "role": "assistant",
                "content": d.get("response", ""),
                "sources": d.get("sources", []),
                "ticket_id": d.get("ticket_id"),
                "message_id": d.get("message_id", uuid.uuid4().hex),
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error ({r.status_code}). Please try again.",
                "message_id": uuid.uuid4().hex,
            })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ {type(e).__name__}: Could not reach server.",
            "message_id": uuid.uuid4().hex,
        })
    st.rerun()


# ─── Voice Input ─────────────────────────────────────────────────────────────
st.divider()
vcol, tcol = st.columns([1, 3])

with vcol:
    st.markdown("**🎤 Voice Input**")
    audio_data = st.audio_input(
        "Record a question",
        key="mic_input",
        on_change=on_audio_change,  # fires ONCE per new recording
    )

with tcol:
    # Transcribe ONLY when the callback set the flag.
    if st.session_state.needs_transcription and audio_data is not None:
        raw = audio_data.getvalue()
        if raw:
            with st.spinner("🎤 Transcribing..."):
                try:
                    r = httpx.post(
                        f"{API_BASE_URL}/voice/transcribe",
                        files={"file": ("recording.wav", raw, "audio/wav")},
                        timeout=httpx.Timeout(connect=10, read=60, write=10, pool=10),
                    )
                    if r.status_code == 200:
                        d = r.json()
                        st.session_state.voice_transcript = d.get("transcript", "")
                        st.caption(f"⏱️ Transcribed in {d.get('processing_time_ms', '?')}ms")
                    else:
                        st.warning("🎤 Transcription failed.")
                except Exception:
                    st.warning("🎤 Could not transcribe audio.")
            st.session_state.needs_transcription = False  # Done, don't repeat.

    # Show editable transcript only if not yet sent.
    if st.session_state.voice_transcript and not st.session_state.transcript_sent:
        st.markdown("**✏️ Review & edit your transcript:**")
        edited = st.text_area(
            "Transcript",
            value=st.session_state.voice_transcript,
            key="transcript_edit",
            label_visibility="collapsed",
        )
        st.caption("💡 You can edit the text above before sending.")
        if st.button("✅ Send transcript", type="primary", use_container_width=True):
            if edited and edited.strip():
                st.session_state.transcript_sent = True
                st.session_state.pending_message = edited.strip()
                st.rerun()

# ─── Text chat input ─────────────────────────────────────────────────────────
if prompt := st.chat_input("How can we help?"):
    st.session_state.pending_message = prompt
    st.rerun()
