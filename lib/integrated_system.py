#!/usr/bin/env python3
"""
Integrated Speech Recognition & Synthesis System

Combines Whisper recognition with AivisSpeech synthesis in a unified system.
"""

import asyncio
import concurrent.futures as cf
from typing import Optional

from .recognizer.whisper_recognizer import WhisperRecognizer
from .synthesizer.synthesizer import AivisSpeechSynthesizer


class IntegratedSpeechSystem:
    """Integrated speech recognition and synthesis system."""

    def __init__(
        self,
        whisper_model: str = "large",
        speaker_id: int = 1431611904,  # まい speaker
        language: str = "ja",
        silence_threshold: float = 0.028,
        auto_synthesize: bool = True,
        volume: float = 1.0,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Initialize integrated speech system.

        Args:
            whisper_model: Whisper model size
            speaker_id: AivisSpeech speaker ID
            language: Recognition language
            silence_threshold: RMS threshold for speech detection
            auto_synthesize: Whether to automatically synthesize recognized text
            volume: Playback volume
            device: Device for Whisper (cpu, cuda)
            compute_type: Compute type for faster-whisper
        """
        self.auto_synthesize = auto_synthesize
        self.volume = volume
        self.speaker_id = speaker_id

        # Initialize recognizer
        self.recognizer = WhisperRecognizer(
            model_name=whisper_model,
            language=language,
            silence_threshold=silence_threshold,
            device=device,
            compute_type=compute_type,
            on_text_callback=self._on_text_recognized if auto_synthesize else None,
        )

        # Synthesizer will be initialized later
        self.synthesizer: Optional[AivisSpeechSynthesizer] = None
        self.vv_client = None

        # Will be set to the main asyncio loop by the runner
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Track pending synthesis tasks scheduled onto the loop
        self._pending_synth_futures: set[cf.Future] = set()

    async def initialize_aivisspeech(self, endpoint: str = "http://localhost:10101"):
        """Initialize AivisSpeech synthesizer."""
        try:
            self.synthesizer = AivisSpeechSynthesizer(
                endpoint=endpoint, speaker_id=self.speaker_id, volume=self.volume
            )
            await self.synthesizer.__aenter__()
            print("🎤 AivisSpeech initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize AivisSpeech: {e}")
            print("💡 Make sure AivisSpeech is running on localhost:10101")
            return False

    def start_recording(self):
        """Start the integrated system."""
        if not self.synthesizer and self.auto_synthesize:
            print(
                "❌ AivisSpeech not initialized. Call initialize_aivisspeech() first."
            )
            return

        self.recognizer.start_recording()
        print("🚀 Integrated Speech System Started!")
        print(f"🎤 AivisSpeech Speaker: {self.speaker_id}")
        print(f"🔊 Auto-synthesis: {'Enabled' if self.auto_synthesize else 'Disabled'}")
        print(f"🔊 Volume: {self.volume}")

    def stop_recording(self):
        """Stop the integrated system."""
        self.recognizer.stop_recording()
        print("🛑 System stopped.")

    def _on_text_recognized(self, text: str, rms: float, duration: float):
        """Callback when text is recognized - schedule synthesis if enabled."""
        if self.auto_synthesize and self.synthesizer and self.loop:
            # Schedule synthesis on the main event loop
            fut = asyncio.run_coroutine_threadsafe(
                self._synthesize_text(text), self.loop
            )
            # Track and auto-remove when done
            self._pending_synth_futures.add(fut)

            def _cleanup(_):
                self._pending_synth_futures.discard(fut)

            fut.add_done_callback(_cleanup)

    async def _synthesize_text(self, text: str):
        """Synthesize recognized text to speech."""
        try:
            if not self.synthesizer:
                return
            await self.synthesizer.synthesize_once(text)
            print(f"    🎤 Synthesized: {text[:50]}{'...' if len(text) > 50 else ''}")
        except Exception as e:
            print(f"❌ Synthesis error: {e}")

    async def close_async(self):
        """Close network resources and cancel any pending tasks."""
        # Cancel any pending synthesis tasks
        for fut in list(self._pending_synth_futures):
            if not fut.done():
                fut.cancel()
        self._pending_synth_futures.clear()

        # Close synthesizer if available
        if self.synthesizer:
            await self.synthesizer.__aexit__(None, None, None)
            self.synthesizer = None

    def get_text_history(self):
        """Get history of recognized text."""
        return self.recognizer.get_text_history()

    def clear_text_history(self):
        """Clear text history."""
        self.recognizer.clear_text_history()

    def set_volume(self, volume: float):
        """Set playback volume."""
        self.volume = volume
        if self.synthesizer:
            self.synthesizer.set_volume(volume)

    def set_silence_threshold(self, threshold: float):
        """Set silence threshold for speech detection."""
        self.recognizer.set_silence_threshold(threshold)
