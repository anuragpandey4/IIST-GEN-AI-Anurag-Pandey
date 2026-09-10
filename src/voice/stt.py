"""Groq Whisper STT adapter.

Uses Groq's ``/openai/v1/audio/transcriptions`` endpoint with the
``whisper-large-v3-turbo`` model.  Accepts audio bytes (WAV, MP3, M4A, etc.)
and returns plain text.  Timeout and retry behaviour mirrors the LLM client
(``max_retries=1``, ``timeout=30``).
"""

from __future__ import annotations

import io
import logging

from groq import AsyncGroq

from src.config import Settings

from .contracts import STTService

logger = logging.getLogger(__name__)

# Map common browser/recorder MIME types to file extensions that the
# Groq Whisper endpoint accepts.
_MIME_TO_EXT: dict[str, str] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac",
}


class GroqSTT(STTService):
    """Speech-to-text via Groq's hosted Whisper model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncGroq | None = None

    async def initialize(self) -> None:
        self._client = AsyncGroq(
            api_key=self._settings.groq_api_key,
            max_retries=1,
            timeout=30.0,
        )
        logger.info("GroqSTT initialized (model=whisper-large-v3-turbo)")

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        if self._client is None:
            raise RuntimeError("GroqSTT has not been initialized")

        ext = _MIME_TO_EXT.get(media_type, "wav")
        filename = f"recording.{ext}"

        # Groq's transcriptions.create() expects a file-like tuple.
        transcription = await self._client.audio.transcriptions.create(
            file=(filename, io.BytesIO(audio_bytes)),
            model="whisper-large-v3-turbo",
            response_format="text",
        )

        return str(transcription).strip()

    async def cleanup(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
