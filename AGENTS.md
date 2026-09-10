# Customer Support Ticket Agent — Agent Continuity File

## Project Overview
This is a Zangoh GenAI assignment implementing a customer support agent with text chat + voice I/O using FastAPI, Streamlit, LangGraph, RAG, and a mock ticket tool.

## Provider Stack
- **LLM**: Groq `qwen/qwen3.8-27b` via `langchain-groq` (`ChatGroq`)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, auto-downloads)
- **Vector DB**: Chroma (local persistent at `.data/vector_db/`)
- **STT**: Groq Whisper `whisper-large-v3-turbo` (reuses Groq API key)
- **TTS**: Edge-TTS `en-US-AriaNeural` (free, no API key)
- **No Docker**: Pure Python, cross-platform (Windows/Linux/macOS)

## Architecture
```
Streamlit → FastAPI → SupportPipeline → LangGraph workflow
                   |                  |→ Chroma/HuggingFace RAG
                   |                  \→ session-bound ticket tool
                   |
                   → VoicePipeline → GroqSTT (Whisper)
                                   → EdgeTTS
```

## Key Files
### Text Chat (Original TODOs — completed)
1. `src/rag/retriever.py` — Chroma init + similarity search (L2 distance)
2. `src/llm/workflow.py` — 4 LangGraph nodes (retrieve, decide, answer, collect_or_create)
3. `src/pipeline.py` — process() orchestration
4. `streamlit_app.py` — UI with text + voice input + speaker playback

### Voice (Mid-Session — completed)
5. `src/voice/contracts.py` — Abstract STTService / TTSService (from supplied code)
6. `src/voice/models.py` — TranscriptionResponse / SynthesisRequest (from supplied code)
7. `src/voice/pipeline.py` — VoicePipeline scaffold (from supplied code)
8. `src/voice/stt.py` — GroqSTT adapter (Groq Whisper, 30s timeout, max_retries=1)
9. `src/voice/tts.py` — EdgeTTS adapter (edge-tts, 30s timeout)

## Design Decisions
- **Relevance threshold 1.5**: L2 distance via Chroma. Configurable via `RAG_RELEVANCE_THRESHOLD`.
- **One Groq API key**: Serves both LLM and Whisper STT — no additional credentials.
- **message_id UUID**: Each ChatResponse gets a stable UUID for TTS audio caching.
- **Voice init is best-effort**: If voice pipeline fails, text chat still works. Voice endpoints return 503.
- **ComponentNotReadyError parity**: Voice endpoints use same error pattern as text pipeline.
- **Idempotency**: TicketRepository prevents duplicate tickets per session.
- **Structured output**: `model.with_structured_output(IntentDecision)` for routing.

## API Key Location
- `.env` file (gitignored) — `GROQ_API_KEY=gsk_...`

## Test Strategy
- 51 tests total (34 text + 17 voice), all use mocks except `test_retriever.py`
- `test_voice.py` — 9 tests: fake adapters, timing, association, regression
- `test_voice_api.py` — 8 tests: endpoint mocking, error codes, 503 handling
- `test_retriever.py` uses real HuggingFace embeddings + temp Chroma dir
