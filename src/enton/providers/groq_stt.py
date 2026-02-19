"""Groq STT Provider — Ultra-low latency speech recognition via LPU.

Uses Groq's cloud API to run Whisper-large-v3 faster than real-time.
Fallback-ready and lightweight (httpx based).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import httpx
import numpy as np

if TYPE_CHECKING:
    from enton.core.config import Settings

logger = logging.getLogger(__name__)


class GroqSTT:
    """Groq API wrapper for Whisper (STT)."""

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.groq_api_key
        self.model = "whisper-large-v3"
        self.endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"
        
        if not self.api_key:
            logger.warning("Groq API Key missing! STT will fail if used.")

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio bytes using Groq API."""
        if not self.api_key:
            return ""

        # Convert float32 numpy array to int16 bytes (PCM)
        # Groq expects a file upload. We'll simulate a WAV file in memory.
        import io
        import wave

        # Denormalize float32 (-1.0 to 1.0) to int16
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        file_content = wav_buffer.read()

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient() as client:
                files = {
                    "file": ("audio.wav", file_content, "audio/wav")
                }
                data = {
                    "model": self.model,
                    "language": "pt", # Force Portuguese for speed
                    "response_format": "json",
                    "temperature": 0.0
                }
                
                headers = {"Authorization": f"Bearer {self.api_key}"}
                
                response = await client.post(
                    self.endpoint, 
                    headers=headers, 
                    data=data, 
                    files=files,
                    timeout=5.0 # Fast timeout
                )
                
                if response.status_code != 200:
                    logger.error("Groq STT Error %d: %s", response.status_code, response.text)
                    return ""

                result = response.json()
                text = result.get("text", "").strip()
                
                lat = (time.perf_counter() - t0) * 1000
                if text:
                    logger.info("Groq STT: '%s' (%.0fms)", text, lat)
                
                return text

        except Exception as e:
            logger.error("Groq STT failed: %s", e)
            return ""

    async def stream(self):
        raise NotImplementedError("Groq does not support streaming STT yet.")
