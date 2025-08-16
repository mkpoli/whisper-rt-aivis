#!/usr/bin/env python3
"""
AivisSpeech Synthesizer Library

Core synthesizer class for text-to-speech using AivisSpeech.
"""

import asyncio
from typing import Optional

import numpy as np
import pyaudio
import vvclient


class AivisSpeechSynthesizer:
    """AivisSpeech text-to-speech synthesizer with volume control."""
    
    def __init__(self, endpoint: str = "http://localhost:10101", speaker_id: int = 1431611904, volume: float = 1.0):
        self.endpoint = endpoint
        self.speaker_id = speaker_id
        self.volume = max(0.0, min(2.0, volume))
        self._client: Optional[vvclient.Client] = None
        self._pa: Optional[pyaudio.PyAudio] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = vvclient.Client(self.endpoint)
        # Touch the engine to verify connectivity
        version = await self._client.fetch_engine_version()
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
        """Play WAV audio data with volume control."""
        assert self._pa is not None, "PyAudio not initialized"
        audio_array = np.frombuffer(wav_data, dtype=np.int16)
        adjusted = (audio_array * volume).astype(np.int16)
        adjusted_bytes = adjusted.tobytes()
        stream = self._pa.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
        try:
            stream.write(adjusted_bytes)
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
