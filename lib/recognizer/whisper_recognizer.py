#!/usr/bin/env python3
"""
Whisper Recognizer Library

Core speech recognition class using OpenAI Whisper.
"""

import time
import threading
from collections import deque
from typing import Optional, Callable, List, Deque, Dict, cast
import os
import sys
from pathlib import Path

import numpy as np
import pyaudio  # type: ignore
from faster_whisper import WhisperModel  # type: ignore


class WhisperRecognizer:
    """Real-time speech recognition using OpenAI Whisper."""

    def __init__(
        self,
        model_name: str = "large",
        language: str = "ja",
        silence_threshold: float = 0.028,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        device: str = "cpu",
        compute_type: str = "int8",
        on_text_callback: Optional[Callable[[str, float, float], None]] = None,
        stop_phrases: Optional[List[str]] = None,
    ):
        """
        Initialize Whisper recognizer.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            language: Recognition language code
            silence_threshold: RMS threshold for speech detection
            sample_rate: Audio sample rate
            chunk_duration: Audio chunk duration in seconds
            device: Device for Whisper (cpu, cuda)
            compute_type: Compute type for faster-whisper
            on_text_callback: Callback function for recognized text
        """
        self.model_name = model_name
        self.language = language
        self.silence_threshold = silence_threshold
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(self.sample_rate * chunk_duration)
        self.device = device
        self.compute_type = compute_type
        self.on_text_callback = on_text_callback

        # Stop-phrases to suppress common spurious outputs (e.g., JP YouTube outro)
        default_stop_phrases = [
            "ご視聴ありがとうございました",
        ]
        self.stop_phrases = set((stop_phrases or []) + default_stop_phrases)

        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_buffer: Deque[np.int16] = deque(maxlen=int(self.sample_rate * 10))
        self.continuous_speech_buffer: Deque[np.int16] = deque(
            maxlen=int(self.sample_rate * 15)
        )
        # Buffer for the current utterance
        self.utterance_buffer: Deque[np.int16] = deque(
            maxlen=int(self.sample_rate * 20)
        )

        # Speech detection
        self.is_speaking = False
        self._prev_is_speaking = False
        self.speech_start_time = 0
        self.continuous_speech_duration = 0
        self.last_speech_time = 0

        # Processing
        self.processing_thread = None
        self.last_recognized_text = ""
        self.text_buffer: List[str] = []

        # Emission and context tuning
        self.min_ctx_seconds: float = (
            4.0  # ensure enough context to avoid mid-sentence cuts
        )
        self.max_ctx_seconds: float = 10.0  # cap to keep latency bounded
        self.use_vad_filter: bool = (
            True  # leverage faster-whisper VAD to stabilize segments
        )
        self._last_emitted_text: str = ""
        self._last_emit_time: float = 0.0
        self._sentence_enders = set(
            [".", "!", "?", "。", "！", "？", "…", "♪", "〜", "～"]
        )
        self._need_finalize_on_eos: bool = False
        # Duplicate suppression
        self.suppress_repeat_seconds: float = 5.0
        self._emitted_text_to_time: Dict[str, float] = {}

        # Initialize Whisper
        print(
            f"🎯 Loading Whisper model: {model_name} [device={device}, compute={compute_type}]"
        )
        if self.device == "cuda" and sys.platform == "win32":
            self._prepare_cuda_dlls()
        self.whisper = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
        print("✅ Whisper model loaded successfully!")

    def _prepare_cuda_dlls(self) -> None:
        """Prepare CUDA DLLs for Windows."""
        try:
            import torch  # noqa: F401

            torch_path = __import__("torch").__file__
            if torch_path is None:
                return
            torch_lib = Path(cast(str, torch_path)).parent / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
                print(f"✅ Added CUDA DLL path: {torch_lib}")
        except Exception as e:
            print(f"⚠️ CUDA DLL prep failed (will try anyway): {e}")

    def start_recording(self):
        """Start recording and processing audio."""
        if self.is_recording:
            print("Already recording!")
            return

        self.is_recording = True
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )
        self.stream.start_stream()
        self.processing_thread = threading.Thread(
            target=self._process_audio, daemon=True
        )
        self.processing_thread.start()
        print("🚀 Whisper Recognizer Started!")
        print(f"🎯 Whisper Model: {self.model_name}")
        print(f"🌍 Language: {self.language}")
        print(f"🔊 Silence Threshold: {self.silence_threshold}")
        print(f"🖥️ Device: {self.device} | ⚙️ Compute: {self.compute_type}")
        print("💬 Speak now! Press Ctrl+C to stop.\n")

    def stop_recording(self):
        """Stop recording and processing."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()
        self.audio.terminate()
        print("🛑 Recognizer stopped.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback for real-time processing."""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_buffer.extend(audio_data)
            self.continuous_speech_buffer.extend(audio_data)
            # Detect speech and track transitions
            prev_state = self._prev_is_speaking
            self._detect_speech(audio_data)
            now_state = self.is_speaking
            if now_state and not prev_state:
                # Start of a new utterance
                self.utterance_buffer.clear()
            if now_state:
                # Accumulate current utterance audio
                self.utterance_buffer.extend(audio_data)
            if prev_state and not now_state:
                # End of speech detected; request finalization in processing loop
                self._need_finalize_on_eos = True
            self._prev_is_speaking = now_state
        return (in_data, pyaudio.paContinue)

    def _detect_speech(self, audio_data):
        """Detect speech based on audio levels."""
        audio_float = audio_data.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float**2))
        now = time.time()

        if rms > self.silence_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = now
                self.continuous_speech_duration = 0
            else:
                self.continuous_speech_duration = now - self.speech_start_time
            self.last_speech_time = now
        else:
            if self.is_speaking:
                silence_duration = now - self.last_speech_time
                silence_threshold = max(1.0, self.continuous_speech_duration * 0.3)
                if silence_duration > silence_threshold:
                    self.is_speaking = False

    def _process_audio(self):
        """Main audio processing loop."""
        while self.is_recording:
            if len(self.audio_buffer) >= self.chunk_size:
                for _ in range(self.chunk_size):
                    if self.audio_buffer:
                        self.audio_buffer.popleft()

                if self.is_speaking:
                    ctx_seconds = max(
                        self.min_ctx_seconds,
                        min(self.continuous_speech_duration, self.max_ctx_seconds),
                    )
                    ctx = self._get_dynamic_context(ctx_seconds)
                    if ctx is not None:
                        try:
                            audio_float = ctx.astype(np.float32) / 32768.0
                            current_rms = np.sqrt(np.mean(audio_float**2))
                            segments, info = self.whisper.transcribe(
                                audio_float,
                                language=self.language,
                                vad_filter=self.use_vad_filter,
                            )
                            seg_list = list(segments)
                            text = " ".join([s.text for s in seg_list]).strip()

                            if self._should_emit(text):
                                self._emit_text(text, seg_list, current_rms)

                        except Exception as e:
                            print(f"❌ Error processing audio: {e}")
                else:
                    if self._need_finalize_on_eos and len(self.utterance_buffer) > 0:
                        try:
                            # Final pass over the full utterance to recover any trailing words
                            utt = np.array(list(self.utterance_buffer), dtype=np.int16)
                            audio_float = utt.astype(np.float32) / 32768.0
                            current_rms = np.sqrt(np.mean(audio_float**2))
                            segments, info = self.whisper.transcribe(
                                audio_float,
                                language=self.language,
                                vad_filter=self.use_vad_filter,
                            )
                            seg_list = list(segments)
                            text = " ".join([s.text for s in seg_list]).strip()
                            if text:
                                self._emit_text(text, seg_list, current_rms, force=True)
                        except Exception as e:
                            print(f"❌ Error in finalization: {e}")
                        finally:
                            self._need_finalize_on_eos = False
                            self.utterance_buffer.clear()

            time.sleep(0.1)

    def _get_speech_context(self, min_duration: float = 2.0):
        """Get audio context for speech processing (legacy)."""
        if not self.is_speaking:
            return None
        target = int(self.sample_rate * min_duration)
        if len(self.continuous_speech_buffer) >= target:
            return np.array(list(self.continuous_speech_buffer)[-target:])
        return None

    def _get_dynamic_context(self, seconds: float):
        """Get a dynamic-length context from the current utterance buffer."""
        if not self.is_speaking:
            return None
        target = int(self.sample_rate * seconds)
        if len(self.utterance_buffer) >= target:
            return np.array(list(self.utterance_buffer)[-target:])
        # If utterance shorter than target, return all we have once it passes a minimal threshold
        if len(self.utterance_buffer) >= int(
            self.sample_rate * self.min_ctx_seconds * 0.5
        ):
            return np.array(list(self.utterance_buffer))
        return None

    def _ends_sentence(self, text: str) -> bool:
        return bool(text) and text[-1] in self._sentence_enders

    def _is_stop_text(self, t: str) -> bool:
        stripped = t.strip()
        for stop in self.stop_phrases:
            if stripped == stop or stripped.endswith(stop):
                return True
        return False

    def _is_recent_duplicate(self, text: str) -> bool:
        now = time.time()
        last_time = self._emitted_text_to_time.get(text)
        return (
            last_time is not None and (now - last_time) < self.suppress_repeat_seconds
        )

    def _should_emit(self, text: str) -> bool:
        if not text or self._is_stop_text(text):
            return False
        if text == self._last_emitted_text:
            return False
        if self._is_recent_duplicate(text):
            return False
        # Prefer sentence-complete outputs
        if self._ends_sentence(text):
            return True
        # Otherwise rate-limit partials and require enough delta to avoid tiny fragments
        now = time.time()
        progressed = len(text) - len(self._last_emitted_text)
        if progressed >= 12 and (now - self._last_emit_time) >= 1.2:
            return True
        return False

    def _emit_text(self, text: str, seg_list, current_rms: float, force: bool = False):
        if not text:
            return
        if self._is_stop_text(text):
            return
        if not force and text == self._last_emitted_text:
            return
        self._last_emitted_text = text
        self.last_recognized_text = text
        now = time.time()
        self._last_emit_time = now
        self._emitted_text_to_time[text] = now

        ts = time.strftime("%H:%M:%S")
        seg_start = (
            seg_list[0].start
            if seg_list and getattr(seg_list[0], "start", None) is not None
            else None
        )
        seg_end = (
            seg_list[-1].end
            if seg_list and getattr(seg_list[-1], "end", None) is not None
            else None
        )
        if seg_start is not None and seg_end is not None and seg_end >= seg_start:
            dur_seconds = float(seg_end - seg_start)
        else:
            dur_seconds = float(max(self.min_ctx_seconds, 2.0))
        dur = f"{dur_seconds:.1f}s"

        print(f"[{ts}] 🎯 {text}")
        print(f"    🔊 Level: {current_rms:.4f} | ⏱️ Duration: {dur}")

        self.text_buffer.append(text)

        if self.on_text_callback:
            self.on_text_callback(text, current_rms, self.continuous_speech_duration)

    def get_text_history(self) -> List[str]:
        """Get history of recognized text."""
        return self.text_buffer.copy()

    def clear_text_history(self):
        """Clear text history."""
        self.text_buffer.clear()
        self.last_recognized_text = ""

    def set_silence_threshold(self, threshold: float):
        """Set silence threshold for speech detection."""
        self.silence_threshold = threshold

    def get_silence_threshold(self) -> float:
        """Get current silence threshold."""
        return self.silence_threshold
