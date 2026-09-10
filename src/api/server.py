from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response

from src.config import load_settings
from src.models import ChatRequest, ChatResponse, Ticket
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from src.voice.contracts import STTService, TTSService
from src.voice.models import TranscriptionResponse, SynthesisRequest
from src.voice.pipeline import VoicePipeline
from src.voice.stt import GroqSTT
from src.voice.tts import GoogleTTS

logger = logging.getLogger(__name__)

settings = load_settings()
pipeline = SupportPipeline(settings, Path(__file__).resolve().parents[2] / "knowledge_base")

# Voice pipeline — uses the same Groq key for Whisper STT + free Edge-TTS.
_stt: STTService = GroqSTT(settings)
_tts: TTSService = GoogleTTS()
voice_pipeline = VoicePipeline(_stt, _tts)
voice_ready = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup owns expensive shared initialization. Request handlers reuse the
    # resulting components, while shutdown makes readiness false immediately.
    global voice_ready
    await pipeline.initialize()

    # Voice init is best-effort: if it fails, text chat still works.
    try:
        await voice_pipeline.initialize()
        voice_ready = True
        logger.info("Voice pipeline initialized (STT + TTS ready)")
    except Exception:
        logger.exception("Voice pipeline failed to initialize — voice endpoints will return 503")
        voice_ready = False

    yield

    pipeline.ready = False
    voice_ready = False
    await voice_pipeline.cleanup()


app = FastAPI(title="Customer Support Ticket Agent", lifespan=lifespan)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    if not pipeline.ready:
        raise HTTPException(status_code=503, detail="Support pipeline is not ready")
    return {"status": "ready", "voice": "ready" if voice_ready else "unavailable"}


# ── Text Chat (unchanged contract) ───────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Keep this transport boundary thin: Pydantic validates the public request,
    # the pipeline owns orchestration, and known service errors are translated
    # to stable HTTP responses here.
    try:
        return await pipeline.process(request.session_id, request.message)
    except ComponentNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ── Tickets (unchanged) ──────────────────────────────────────────────────────

@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: str) -> Ticket:
    # Keep this endpoint read-only. It should return the exact repository record,
    # not ask the model to reconstruct ticket details. Test both the successful
    # lookup and unknown-ID response through FastAPI's test client.
    ticket = pipeline.tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


# ── Voice: Transcribe (STT) ──────────────────────────────────────────────────

@app.post("/voice/transcribe", response_model=TranscriptionResponse)
async def voice_transcribe(file: UploadFile = File(...)):
    """Accept an audio file upload and return its transcript."""
    if not voice_ready:
        raise HTTPException(
            status_code=503,
            detail="Voice pipeline is not ready",
        )

    audio_bytes = await file.read()
    media_type = file.content_type or "audio/wav"

    try:
        transcript, processing_time_ms = await voice_pipeline.transcribe(
            audio_bytes, media_type
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("STT transcription failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Speech-to-text transcription failed. Please try again.",
        ) from exc

    return TranscriptionResponse(
        success=True,
        transcript=transcript,
        processing_time_ms=processing_time_ms,
    )


# ── Voice: Synthesize (TTS) ──────────────────────────────────────────────────

@app.post("/voice/synthesize")
async def voice_synthesize(request: SynthesisRequest):
    """Accept response text and return playable audio bytes."""
    if not voice_ready:
        raise HTTPException(
            status_code=503,
            detail="Voice pipeline is not ready",
        )

    try:
        audio_bytes, media_type, _ = await voice_pipeline.synthesize(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("TTS synthesis failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Text-to-speech synthesis failed. Please try again.",
        ) from exc

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{request.message_id}.mp3"'},
    )
