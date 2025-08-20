#!/usr/bin/env python3
"""
AivisSpeech Synthesizer Library

Core synthesizer class for text-to-speech using AivisSpeech.
"""

import asyncio
from typing import Optional

import numpy as np
import pyaudio
import io
import wave
import vvclient


class AivisSpeechSynthesizer:
    """AivisSpeech text-to-speech synthesizer with volume control."""

    def __init__(
        self,
        endpoint: str = "http://localhost:10101",
        speaker_id: int = 888753760,
        volume: float = 1.0,
    ):
        self.endpoint = endpoint
        self.speaker_id = speaker_id
        self.volume = max(0.0, min(2.0, volume))
        self._client: Optional[vvclient.Client] = None
        self._pa: Optional[pyaudio.PyAudio] = None

    async def __aenter__(self):
        """Async context manager entry."""
        client = vvclient.Client(self.endpoint)
        self._client = client
        # Touch the engine to verify connectivity
        version = await client.fetch_engine_version()
        print(f"🎤 AivisSpeech connected: {version}")
        self._pa = pyaudio.PyAudio()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit."""
        if self._pa:
            self._pa.terminate()
            self._pa = None
        # vvclient.Client supports async context normally; close session if available
        try:
            if self._client:
                await self._client.close()
        except Exception:
            pass
        self._client = None

    async def synthesize_once(self, text: str):
        """Synthesize a single piece of text to speech."""
        if not text or not text.strip():
            return
        assert self._client is not None, "Client not initialized"
        # Create query and synthesize
        query = await self._client.create_audio_query(text, speaker=self.speaker_id)
        wav_data = await query.synthesis(speaker=self.speaker_id)
        self._play_wav(wav_data, self.volume)

    def _play_wav(self, wav_data: bytes, volume: float):
        """Play WAV audio data with volume control and graceful fade-in."""
        assert self._pa is not None, "PyAudio not initialized"

        # Decode WAV container to raw PCM to avoid header artifacts
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            num_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()
            frames_bytes = wf.readframes(num_frames)

        if sample_width != 2:
            # Only int16 is supported here; other widths would need conversion
            raise RuntimeError(f"Unsupported WAV sample width: {sample_width*8} bits")

        # Convert to int16 PCM array
        pcm_array = np.frombuffer(frames_bytes, dtype=np.int16).copy()

        # Apply volume scaling
        if volume != 1.0:
            pcm_array = (
                (pcm_array.astype(np.float32) * float(volume))
                .clip(-32768, 32767)
                .astype(np.int16)
            )

        # Apply a short fade-in (10 ms) to avoid clicks at start
        fade_in_duration_seconds = 0.01
        fade_samples = int(fade_in_duration_seconds * sample_rate)
        fade_samples = max(
            0, min(fade_samples, pcm_array.shape[0] // max(1, num_channels))
        )
        if fade_samples > 0:
            if num_channels > 1:
                # Reshape to (frames, channels) to apply ramp per frame
                pcm_frames = pcm_array.reshape(-1, num_channels).astype(np.float32)
                ramp = np.linspace(
                    0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32
                )
                pcm_frames[:fade_samples, :] *= ramp[:, None]
                pcm_array = pcm_frames.astype(np.int16).reshape(-1)
            else:
                pcm_float = pcm_array.astype(np.float32)
                ramp = np.linspace(
                    0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32
                )
                pcm_float[:fade_samples] *= ramp
                pcm_array = pcm_float.astype(np.int16)

        # Open stream with proper format
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=num_channels,
            rate=sample_rate,
            output=True,
        )
        try:
            stream.write(pcm_array.tobytes())
        finally:
            stream.stop_stream()
            stream.close()

    def set_volume(self, volume: float):
        """Set playback volume (0.0 to 2.0)."""
        self.volume = max(0.0, min(2.0, volume))

    def get_speaker_id(self) -> int:
        """Get current speaker ID."""
        return self.speaker_id

    def set_speaker_id(self, speaker_id: int):
        """Set speaker ID for synthesis."""
        self.speaker_id = speaker_id
