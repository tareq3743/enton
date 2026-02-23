from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import torch

from enton.core.blob_store import BlobType
from enton.core.config import Provider
from enton.core.events import EventBus, TranscriptionEvent
from enton.providers.google import GoogleSTT
from enton.providers.groq_stt import GroqSTT
from enton.providers.local import LocalSTT
from enton.providers.nvidia import NvidiaSTT

if TYPE_CHECKING:
    from enton.core.config import Settings
    from enton.providers.base import STTProvider

logger = logging.getLogger(__name__)

# Streaming config
_PARTIAL_INTERVAL_CHUNKS = 48  # ~1.5s @ 512 samples / 16kHz
_MIN_PARTIAL_AUDIO = 0.5  # seconds — don't transcribe partials shorter than this


class Ears:
    # Priority order for fallback (Fastest Cloud -> Reliable Cloud -> Local)
    _FALLBACK_ORDER = [
        Provider.GROQ,
        Provider.NVIDIA,
        Provider.GOOGLE,
        Provider.LOCAL,
    ]

    def __init__(self, settings: Settings, bus: EventBus, blob_store: object | None = None) -> None:
        self._settings = settings
        self._bus = bus
        self._blob_store = blob_store
        self._providers: dict[Provider, STTProvider] = {}
        self._primary = settings.stt_provider
        self._muted = False
        self._init_providers(settings)

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        self._muted = value

    def _init_providers(self, s: Settings) -> None:
        if s.groq_api_key:
            try:
                self._providers[Provider.GROQ] = GroqSTT(s)
            except Exception:
                logger.warning("Groq STT unavailable")

        if s.nvidia_api_key:
            try:
                NvidiaSTT_instance = NvidiaSTT(s)
                self._providers[Provider.NVIDIA] = NvidiaSTT_instance
            except Exception:
                logger.warning("NVIDIA STT unavailable")

        if s.google_project:
            try:
                GoogleSTT_instance = GoogleSTT(s)
                self._providers[Provider.GOOGLE] = GoogleSTT_instance
            except Exception:
                logger.warning("Google STT unavailable")

        if s.whisper_model:
            try:
                LocalSTT_instance = LocalSTT(s)
                self._providers[Provider.LOCAL] = LocalSTT_instance
            except Exception:
                logger.warning("Local STT unavailable")

    async def transcribe(self, audio: np.ndarray) -> str:
        # 1. Build list of providers to try (Primary -> Fallbacks)
        candidates = [self._primary] + [
            p for p in self._FALLBACK_ORDER if p != self._primary
        ]
        
        # 2. Try each provider in sequence
        for provider_id in candidates:
            if provider_id not in self._providers:
                continue
                
            provider = self._providers[provider_id]
            try:
                # Timeout for cloud providers to fail fast
                if provider_id != Provider.LOCAL:
                    text = await asyncio.wait_for(
                        provider.transcribe(audio, self._settings.sample_rate),
                        timeout=4.0
                    )
                else:
                    text = await provider.transcribe(audio, self._settings.sample_rate)

                if text and text.strip():
                    await self._bus.emit(TranscriptionEvent(text=text, is_final=True))
                    logger.info("Ears [%s]: %s", provider_id, text[:80])
                    return text
                    
            except Exception as e:
                logger.warning("STT [%s] failed: %s. Trying next...", provider_id, e)
                continue

        logger.error("All STT providers failed.")
        return ""

    async def _transcribe_partial(self, audio: np.ndarray) -> str:
        """Transcribe partial audio (during speech) — no fallback, fast path."""
        name, provider = self._get_provider()
        try:
            text = await provider.transcribe(audio, self._settings.sample_rate)
            if text.strip():
                await self._bus.emit(TranscriptionEvent(text=text, is_final=False))
                logger.debug("Ears partial [%s]: %s", name, text[:60])
            return text
        except Exception:
            logger.debug("Partial transcription failed")
            return ""

    async def _save_audio(self, audio: np.ndarray) -> None:
        """Save audio segment to BlobStore as WAV."""
        try:
            buf = io.BytesIO()
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._settings.sample_rate)
                wf.writeframes(pcm.tobytes())

            await self._blob_store.store(
                buf.getvalue(),
                BlobType.AUDIO,
                extension=".wav",
                tags=["speech"],
            )
        except Exception:
            logger.debug("Failed to save audio blob")

    async def run(self) -> None:
        """Continuous mic capture loop with VAD + streaming partial transcription."""
        # logger.info("Loading silero-vad...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )

        # Audio Device Check
        try:
            device_info = sd.query_devices(kind="input")
            device_name = device_info.get("name", "Unknown")
            logger.info("Ears initialized on device: %s", device_name)
            # Log all inputs for debug if needed
            # logger.debug("Available inputs: %s", sd.query_devices())
        except Exception:
            logger.warning("Could not query audio devices")

        window_size_samples = 512
        pre_buffer_chunks = 6  # ~200ms of audio before speech trigger

        logger.info("Ears listening (sample_rate=%d, VAD+streaming)", self._settings.sample_rate)

        queue: asyncio.Queue = asyncio.Queue()

        def _callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio input status: %s", status)
            queue.put_nowait(indata.copy())

        # Audio buffer state
        buffer: list[np.ndarray] = []
        pre_buffer: list[np.ndarray] = []  # rolling buffer for pre-speech context
        is_speaking = False
        silence_counter = 0
        silence_threshold = 20  # ~0.6s of silence to end speech
        max_buffer = 500  # ~16s max
        chunks_since_partial = 0  # counter for partial transcription intervals

        stream = sd.InputStream(
            samplerate=self._settings.sample_rate,
            blocksize=window_size_samples,
            channels=1,
            dtype=np.float32,
            callback=_callback,
        )

        with stream:
            while True:
                chunk = await queue.get()
                chunk = chunk.squeeze()

                if self._muted:
                    buffer.clear()
                    pre_buffer.clear()
                    is_speaking = False
                    chunks_since_partial = 0
                    continue

                chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
                speech_prob = model(chunk_tensor, self._settings.sample_rate).item()

                if speech_prob > 0.5:
                    if not is_speaking:
                        # Speech just started — prepend pre-buffer for context
                        is_speaking = True
                        buffer.extend(pre_buffer)
                        pre_buffer.clear()
                        chunks_since_partial = 0
                    silence_counter = 0
                    buffer.append(chunk)
                    chunks_since_partial += 1

                    # Streaming: emit partial transcription periodically
                    if chunks_since_partial >= _PARTIAL_INTERVAL_CHUNKS:
                        audio_so_far = np.concatenate(buffer)
                        duration = len(audio_so_far) / self._settings.sample_rate
                        if duration >= _MIN_PARTIAL_AUDIO:
                            asyncio.create_task(self._transcribe_partial(audio_so_far))
                        chunks_since_partial = 0

                elif is_speaking:
                    buffer.append(chunk)
                    silence_counter += 1
                    if silence_counter > silence_threshold:
                        is_speaking = False
                        full_audio = np.concatenate(buffer)
                        buffer.clear()
                        silence_counter = 0
                        chunks_since_partial = 0
                        if len(full_audio) > self._settings.sample_rate * 0.5:
                            logger.info(
                                "Speech detected (%.2fs)",
                                len(full_audio) / self._settings.sample_rate,
                            )
                            asyncio.create_task(self.transcribe(full_audio))
                            if self._blob_store:
                                asyncio.create_task(self._save_audio(full_audio))
                else:
                    # Not speaking — maintain rolling pre-buffer
                    pre_buffer.append(chunk)
                    if len(pre_buffer) > pre_buffer_chunks:
                        pre_buffer.pop(0)

                # Safety valve
                if len(buffer) > max_buffer:
                    logger.warning("Audio buffer overflow, flushing")
                    is_speaking = False
                    full_audio = np.concatenate(buffer)
                    buffer.clear()
                    chunks_since_partial = 0
                    asyncio.create_task(self.transcribe(full_audio))
