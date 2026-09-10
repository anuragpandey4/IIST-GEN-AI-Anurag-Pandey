"""Google TTS (gTTS) adapter.

Uses Google's free Translate TTS endpoint via the ``gTTS`` package.
Requires internet access. No API key needed.

Output: MP3 audio bytes (``audio/mpeg``).
Timeout: 30 s via ``asyncio.wait_for()``.
"""

from __future__ import annotations

import asyncio
import io
import logging

from gtts import gTTS

from .contracts import TTSService

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30


class GoogleTTS(TTSService):
    """Text-to-speech via Google Translate TTS (free, no API key)."""

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang

    async def initialize(self) -> None:
        logger.info("GoogleTTS initialized (lang=%s)", self._lang)

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        def _generate() -> bytes:
            tts = gTTS(text=text, lang=self._lang)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            return buffer.getvalue()

        # Run blocking gTTS call in a thread to avoid blocking the event loop.
        audio_bytes = await asyncio.wait_for(
            asyncio.to_thread(_generate),
            timeout=_TIMEOUT_SECONDS,
        )

        if not audio_bytes:
            raise RuntimeError("gTTS returned empty audio")

        return audio_bytes, "audio/mpeg"

    async def cleanup(self) -> None:
        pass  # gTTS is stateless.
