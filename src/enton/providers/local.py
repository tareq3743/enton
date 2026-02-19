from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from faster_whisper import WhisperModel
from kokoro import KPipeline

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from enton.core.config import Settings

logger = logging.getLogger(__name__)


class LocalSTT:
    """faster-whisper local STT fallback."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.whisper_model
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = WhisperModel(
                self._model_name,
                device="cuda",
                compute_type="float16",
            )
        return self._model

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        model = self._ensure_model()
        loop = asyncio.get_running_loop()

        def _transcribe():
            segments, _ = model.transcribe(audio, language="pt", beam_size=5)
            return " ".join(seg.text for seg in segments)

        return await loop.run_in_executor(None, _transcribe)

    async def stream(self) -> AsyncIterator[str]:
        raise NotImplementedError("Local STT streaming — Phase 2")


class LocalTTS:
    """Kokoro local TTS fallback (GPU Accelerated)."""

    def __init__(self, settings: Settings) -> None:
        self._lang = settings.kokoro_lang
        self._default_voice = settings.kokoro_voice
        self._pipeline = None
        self.sample_rate: int = 24000
        # Check for CUDA
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_pipeline(self):
        if self._pipeline is None:
            logger.info("Loading Kokoro TTS on %s...", self._device)
            self._pipeline = KPipeline(lang_code=self._lang, repo_id="hexgrad/Kokoro-82M")
            if hasattr(self._pipeline, "model") and self._pipeline.model is not None:
                self._pipeline.model = self._pipeline.model.to(torch.device(self._device))
        return self._pipeline

    async def synthesize(self, text: str, voice: str | None = None, speed: float = 1.0) -> np.ndarray:
        """Synthesize full audio."""
        pipeline = self._ensure_pipeline()
        voice_to_use = voice or self._default_voice
        loop = asyncio.get_running_loop()

        def _synth():
            chunks = []
            for _, _, audio in pipeline(text, voice=voice_to_use, speed=speed):
                chunks.append(audio)
            if not chunks:
                return np.array([], dtype=np.float32)
            return np.concatenate(chunks)

        return await loop.run_in_executor(None, _synth)

    async def synthesize_stream(self, text: str, voice: str | None = None, speed: float = 1.0) -> AsyncIterator[np.ndarray]:
        """Yield audio chunks in real-time."""
        pipeline = self._ensure_pipeline()
        voice_to_use = voice or self._default_voice
        loop = asyncio.get_running_loop()
        
        # Generator wrapper to run in thread
        queue = asyncio.Queue()
        
        def _producer():
            try:
                for _, _, audio in pipeline(text, voice=voice_to_use, speed=speed):
                    loop.call_soon_threadsafe(queue.put_nowait, audio)
                loop.call_soon_threadsafe(queue.put_nowait, None) # Sentinel
            except Exception as e:
                logger.error("TTS Stream error: %s", e)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _producer)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
