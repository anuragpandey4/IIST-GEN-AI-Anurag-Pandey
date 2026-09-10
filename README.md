# Customer Support Ticket Agent

An AI-powered customer support agent built with **FastAPI**, **Streamlit**, **LangGraph**, and **Groq**. The agent answers policy questions using RAG (Retrieval-Augmented Generation) and collects ticket information through a multi-turn conversation flow. Includes **voice input** (STT via Groq Whisper) and **response playback** (TTS via Edge-TTS).

## Architecture

```text
┌──────────────────────┐     ┌──────────────────────┐     ┌───────────────────────────────┐
│     Streamlit UI     │     │       FastAPI         │     │      SupportPipeline          │
│     (:8501)          │     │       (:8000)         │     │                               │
│                      │     │                      │     │  ┌─────────────────────────┐  │
│  ┌──────────────┐   │     │  POST /chat ──────────┼────▶│  │   LangGraph Workflow    │  │
│  │ 💬 Text Chat │───┼────▶│                      │     │  │                         │  │
│  └──────────────┘   │     │                      │     │  │  retrieve → decide ─┐   │  │
│                      │◀────│                      │◀────│  │         answer ◀────┤   │  │
│  ┌──────────────┐   │     │                      │     │  │         ticket ◀────┘   │  │
│  │ 🎤 Voice In  │───┼────▶│  POST /voice/transcr │     │  └──────────┬──────────────┘  │
│  │ (audio_input)│   │     │  POST /voice/synthes │     │             │                  │
│  └──────────────┘   │     │                      │     │  ┌──────────┴──────────┐      │
│                      │     │                      │     │  │  Chroma   │ Session │      │
│  ┌──────────────┐   │     └──────────────────────┘     │  │  (RAG)    │ Store   │      │
│  │ 🔊 Speaker   │◀──┼─── audio/mpeg response           │  └───────────┴─────────┘      │
│  │    Icons     │   │                                   └───────────────────────────────┘
│  └──────────────┘   │     ┌──────────────────────┐
│                      │     │    VoicePipeline      │
└──────────────────────┘     │  ┌─────┐  ┌───────┐  │
                              │  │ STT │  │  TTS  │  │
                              │  │Groq │  │ Edge  │  │
                              │  │Whisp│  │  TTS  │  │
                              │  └─────┘  └───────┘  │
                              └──────────────────────┘
```

### Workflow Flow

1. **Retrieve** — Searches the vector knowledge base (Chroma) for relevant policy documents
2. **Decide** — Uses structured LLM output to classify intent as `answer` (policy question) or `ticket` (support request)
3. **Answer** — Generates a grounded response using only retrieved context; refuses to fabricate information
4. **Ticket** — Collects customer details over multiple turns, validates via Pydantic, creates ticket with idempotency

## Tech Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| **LLM** | [Groq](https://console.groq.com) | `qwen/qwen3.8-27b` — open-source model, free tier |
| **Embeddings** | HuggingFace | `sentence-transformers/all-MiniLM-L6-v2` — local, auto-downloads ~80MB |
| **Vector DB** | Chroma | Local persistent storage, rebuilds from `knowledge_base/` on startup |
| **Agent Framework** | LangGraph + LangChain | Typed state graph with conditional routing |
| **STT** | Groq Whisper | `whisper-large-v3-turbo` — reuses existing Groq API key |
| **TTS** | Edge-TTS | Microsoft's free TTS — no API key, `en-US-AriaNeural` voice |
| **API** | FastAPI | Async REST API with Pydantic validation |
| **UI** | Streamlit | Chat + voice input (`st.audio_input`) + speaker playback |

## Prerequisites

- **Python 3.11 or newer**
- **Groq API key** (free): Sign up at [console.groq.com](https://console.groq.com), create an API key
- ~500MB disk space for Python packages + embedding model cache

No GPU, Docker, or local model server required.

## Setup

### Windows (PowerShell)

```powershell
# 1. Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Configure environment
Copy-Item .env.example .env
# Edit .env and replace GROQ_API_KEY with your actual key
```

### Linux (Bash)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and replace GROQ_API_KEY with your actual key
nano .env
```

### macOS (zsh / Bash)

```bash
# Same as Linux — tested with both zsh and bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env and replace GROQ_API_KEY with your actual key
```

> **Note:** On first startup, the HuggingFace embedding model (~80MB) will download automatically to `~/.cache/huggingface/`. Subsequent starts use the cache and are instant.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key. Get one free at [console.groq.com](https://console.groq.com) |
| `LLM_MODEL` | `qwen/qwen3.8-27b` | Groq model name. Other options: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model for RAG |
| `VECTOR_DB_PATH` | `.data/vector_db` | Local directory for Chroma persistence |
| `RAG_COLLECTION` | `customer-support` | Chroma collection name |
| `RAG_TOP_K` | `3` | Maximum number of retrieved chunks per query |
| `RAG_RELEVANCE_THRESHOLD` | `1.5` | Maximum L2 distance for retrieved chunks. Lower = stricter (fewer but more precise results); higher = more lenient (more results but noisier). 1.5 is a balanced default for MiniLM-L6-v2 embeddings with Chroma's L2 distance metric. |
| `API_HOST` | `127.0.0.1` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI port |
| `STREAMLIT_HOST` | `127.0.0.1` | Streamlit bind address |
| `STREAMLIT_PORT` | `8501` | Streamlit port |

## Running

Start the **API server** in one terminal:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8000

# Linux / macOS
source .venv/bin/activate
uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8000
```

Start the **Streamlit UI** in another terminal:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501

# Linux / macOS
source .venv/bin/activate
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

- **UI**: [http://localhost:8501](http://localhost:8501)
- **API docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health check**: [http://localhost:8000/health](http://localhost:8000/health)

### Quick API Test

```bash
# Health check
curl http://localhost:8000/health

# Chat request
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","message":"How long does standard shipping take?"}'
```

For Windows PowerShell:
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health

Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post `
  -ContentType 'application/json' `
  -Body '{"session_id":"demo-1","message":"How long does standard shipping take?"}'
```

## Testing

```bash
# Run all tests
pytest -q

# Run specific test suites
pytest tests/test_session.py -v      # Session state management
pytest tests/test_tickets.py -v      # Ticket repository + validation
pytest tests/test_retriever.py -v    # RAG retrieval (uses real embeddings)
pytest tests/test_workflow.py -v     # LangGraph nodes (mocked LLM)
pytest tests/test_api.py -v          # FastAPI endpoints (mocked pipeline)
pytest tests/test_pipeline.py -v     # Pipeline orchestration
pytest tests/test_voice.py -v        # Voice pipeline (STT/TTS, fake adapters)
pytest tests/test_voice_api.py -v    # Voice API endpoints (mocked pipeline)
```

> **Note:** Tests in `test_retriever.py` download the embedding model on first run (~80MB). All other tests use mocks and require no network access or API keys.

## Voice Features (Mid-Session Requirement)

### Voice Input (Speech-to-Text)
- Click the **🎤 Record a question** microphone button to record audio
- Audio is sent to Groq's Whisper model (`whisper-large-v3-turbo`) for transcription
- The transcript appears in an **editable text area** — you can correct errors before sending
- Click **✅ Send transcript** to submit through the existing text chat pipeline
- STT failures show a warning without breaking the text chat

### Response Playback (Text-to-Speech)
- Every assistant response has a **🔊** speaker button
- Click it to generate and play audio using Edge-TTS (`en-US-AriaNeural` voice)
- Audio is **cached per message** — clicking again replays without re-synthesis
- The button is **disabled during synthesis** to prevent duplicate requests
- TTS failures show a warning but preserve the text response

### Voice API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/voice/transcribe` | POST | Upload audio file → get transcript |
| `/voice/synthesize` | POST | Send text → get MP3 audio bytes |

**Transcription request** (multipart/form-data):
```bash
curl -X POST http://localhost:8000/voice/transcribe -F "file=@recording.wav"
```

**Transcription response:**
```json
{
  "success": true,
  "transcript": "My payment was charged twice",
  "processing_time_ms": 820
}
```

**Synthesis request:**
```bash
curl -X POST http://localhost:8000/voice/synthesize \
  -H "Content-Type: application/json" \
  -d '{"message_id": "abc123", "text": "Shipping takes 3-5 business days."}'
```
Returns: `audio/mpeg` bytes (MP3).

### Provider Choices

| Provider | Why Chosen |
|----------|-----------|
| **Groq Whisper** (STT) | Reuses the existing Groq API key — one key serves both LLM and STT. Direct API call, not an end-to-end platform. |
| **Edge-TTS** (TTS) | Free, no API key required. Explicitly whitelisted in the requirements. High-quality Microsoft voices. |

> **Note:** Both STT and TTS require internet access. No offline claims — Groq Whisper calls Groq's API, Edge-TTS calls Microsoft's endpoint.

## Design Decisions

### Why Groq?
The assignment requires an "OpenAI-compatible endpoint serving an open-source instruction model." Groq serves `qwen/qwen3.8-27b` (open-source) via an OpenAI-compatible API with extremely low latency (~200ms). The free tier is sufficient for development and evaluation. **One Groq API key serves both the LLM and Whisper STT** — no additional credentials needed.

### Why Local Embeddings + Chroma?
Keeping embeddings (HuggingFace) and the vector DB (Chroma) local means:
- No additional API keys for the evaluator
- Works offline after initial model download
- Evaluator can run everything with just one Groq API key

### Relevance Threshold (1.5)
Chroma uses L2 (Euclidean) distance internally. Lower distance means higher similarity. A threshold of 1.5 filters out chunks with L2 distance > 1.5, which represent weak semantic overlap. The threshold is configurable via `RAG_RELEVANCE_THRESHOLD` in `.env` for tuning without code changes.

### Structured Output for Routing
The `decide` node uses `model.with_structured_output(IntentDecision)` to produce a Pydantic-validated `route` field. This makes routing deterministic and testable — no regex parsing of free-text LLM output.

### Category Validation
`TicketCreate` uses Pydantic's `Literal["order", "payment", "account", "technical", "other"]` type. Invalid categories (e.g., "billing") are rejected at validation time. If the LLM extracts an invalid category, the agent asks the customer to choose from the allowed list.

### Ticket Idempotency
`TicketRepository` tracks one ticket per session via `_session_ticket`. Repeated create calls return the original ticket — preventing duplicates from retries or re-submissions.

## Knowledge Base

The agent's RAG knowledge comes from four Markdown policy documents in `knowledge_base/`:

| Document | Topic |
|----------|-------|
| `shipping.md` | Standard (3-5 days) and express (1-2 days) delivery timelines |
| `returns.md` | 30-day return window, refund processing (5-7 business days) |
| `payments.md` | Pending authorizations, duplicate charge reporting |
| `accounts.md` | Password reset, account compromise procedures |

## Troubleshooting

### "GROQ_API_KEY is missing or still set to the placeholder"
Edit `.env` and replace the placeholder with your actual Groq API key from [console.groq.com](https://console.groq.com).

### First startup is slow
The HuggingFace embedding model (~80MB) downloads on first run. Subsequent starts are instant.

### "Cannot connect to the support service"
Make sure the FastAPI server is running in a separate terminal before opening Streamlit.

### Groq rate limit errors
The free tier has rate limits. If you hit them, wait a few seconds between requests or consider a paid plan.

### Port conflicts
Change `API_PORT` or `STREAMLIT_PORT` in `.env` if ports 8000/8501 are already in use.

### Chroma permission errors on Windows
The `.data/vector_db/` directory is created automatically. If you see permission errors, ensure the directory is writable or delete `.data/` and restart.

## Project Structure

```
customer_support_ticket_agent/
├── src/
│   ├── api/
│   │   └── server.py          # FastAPI routes (/health, /chat, /tickets)
│   ├── llm/
│   │   ├── client.py          # Groq ChatGroq adapter
│   │   ├── prompts.py         # System, answer, decide, ticket prompts
│   │   └── workflow.py        # LangGraph state graph (4 nodes + routing)
│   ├── rag/
│   │   ├── document_loader.py # Markdown loading + text splitting
│   │   ├── embeddings.py      # HuggingFace embedding adapter
│   │   └── retriever.py       # Chroma init + similarity search
│   ├── sessions/
│   │   └── store.py           # In-memory conversation state
│   ├── tools/
│   │   └── ticket_tool.py     # Ticket repository + LangChain tool
│   ├── utils/
│   │   └── errors.py          # Custom exception classes
│   ├── config.py              # Environment-based settings
│   ├── models.py              # Pydantic request/response/ticket models
│   └── pipeline.py            # Top-level orchestration
├── tests/
│   ├── test_api.py            # FastAPI endpoint tests
│   ├── test_pipeline.py       # Pipeline orchestration tests
│   ├── test_retriever.py      # RAG retrieval tests
│   ├── test_session.py        # Session state tests
│   ├── test_tickets.py        # Ticket repository + validation tests
│   └── test_workflow.py       # LangGraph workflow tests
├── knowledge_base/            # Support policy documents (4 markdown files)
├── streamlit_app.py           # Chat UI
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
└── README.md                  # This file
```
#   I I S T - G E N - A I - A n u r a g - P a n d e y  
 