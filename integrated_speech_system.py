#!/usr/bin/env python3
"""
Integrated Speech Recognition & Synthesis System

This system combines real-time Whisper speech recognition with AivisSpeech voice synthesis,
creating a complete speech-to-speech pipeline.
"""

import asyncio
import time
import threading
from collections import deque
import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import vvclient

class IntegratedSpeechSystem:
    def __init__(self, 
                 whisper_model="large",
                 speaker_id=1431611904,  # まい speaker
                 language="ja",
                 silence_threshold=0.028,
                 auto_synthesize=True,
                 volume=1.0,
                 device: str = "cpu",
                 compute_type: str = "int8"):
        self.whisper_model = whisper_model
        self.speaker_id = speaker_id
        self.language = language
        self.silence_threshold = silence_threshold
        self.auto_synthesize = auto_synthesize
        self.volume = volume
        self.sample_rate = 16000
        self.chunk_duration = 2.0
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.device = device
        self.compute_type = compute_type
        print(f"🎯 Loading Whisper model: {whisper_model} [device={self.device}, compute={self.compute_type}]")
        if self.device == "cuda" and sys.platform == "win32":
            self._prepare_cuda_dlls()
        self.whisper = WhisperModel(whisper_model, device=self.device, compute_type=self.compute_type)
        print("✅ Whisper model loaded successfully!")
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 10))
        self.continuous_speech_buffer = deque(maxlen=int(self.sample_rate * 15))
        self.is_speaking = False
        self.speech_start_time = 0
        self.continuous_speech_duration = 0
        self.last_speech_time = 0
        self.processing_thread = None
        self.last_recognized_text = ""
        self.text_buffer = []
        self.vv_client = None
        # Will be set to the main asyncio loop by the runner
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Track pending synthesis tasks scheduled onto the loop
        self._pending_synth_futures: set[asyncio.Future] = set()

    async def initialize_aivisspeech(self, endpoint: str = "http://localhost:10101"):
        try:
            self.vv_client = vvclient.Client(endpoint)
            version = await self.vv_client.fetch_engine_version()
            print(f"🎤 AivisSpeech initialized - Version: {version}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize AivisSpeech: {e}")
            print("💡 Make sure AivisSpeech is running on localhost:10101")
            return False

    def start_recording(self):
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
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
        self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.processing_thread.start()
        print("🚀 Integrated Speech System Started!")
        print(f"🎯 Whisper Model: {self.whisper_model}")
        print(f"🎤 AivisSpeech Speaker: {self.speaker_id}")
        print(f"🌍 Language: {self.language}")
        print(f"🔊 Auto-synthesis: {'Enabled' if self.auto_synthesize else 'Disabled'}")
        print(f"🔊 Volume: {self.volume}")
        print("💬 Speak now! Press Ctrl+C to stop.\n")

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()
        self.audio.terminate()
        print("🛑 System stopped.")

    def _prepare_cuda_dlls(self) -> None:
        try:
            import torch  # noqa: F401
            torch_lib = Path(__import__("torch").__file__).parent / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
                print(f"✅ Added CUDA DLL path: {torch_lib}")
        except Exception as e:
            print(f"⚠️ CUDA DLL prep failed (will try anyway): {e}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_buffer.extend(audio_data)
            self.continuous_speech_buffer.extend(audio_data)
            self._detect_speech(audio_data)
        return (in_data, pyaudio.paContinue)

    def _detect_speech(self, audio_data):
        audio_float = audio_data.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
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
        while self.is_recording:
            if len(self.audio_buffer) >= self.chunk_size:
                for _ in range(self.chunk_size):
                    if self.audio_buffer:
                        self.audio_buffer.popleft()
                if self.is_speaking:
                    ctx = self._get_speech_context(min_duration=2.0)
                    if ctx is not None:
                        try:
                            audio_float = ctx.astype(np.float32) / 32768.0
                            current_rms = np.sqrt(np.mean(audio_float ** 2))
                            segments, info = self.whisper.transcribe(audio_float, language=self.language)
                            text = " ".join([s.text for s in segments]).strip()
                            if text and text != self.last_recognized_text:
                                self.last_recognized_text = text
                                ts = time.strftime("%H:%M:%S")
                                dur = f"{self.continuous_speech_duration:.1f}s"
                                print(f"[{ts}] 🎯 {text}")
                                print(f"    🔊 Level: {current_rms:.4f} | ⏱️ Duration: {dur}")
                                self.text_buffer.append(text)
                                if self.auto_synthesize and self.vv_client and self.loop:
                                    # Schedule synthesis on the main event loop
                                    fut = asyncio.run_coroutine_threadsafe(
                                        self._synthesize_text(text), self.loop
                                    )
                                    # Track and auto-remove when done
                                    self._pending_synth_futures.add(fut)
                                    def _cleanup(_):
                                        self._pending_synth_futures.discard(fut)
                                    fut.add_done_callback(_cleanup)
                        except Exception as e:
                            print(f"❌ Error processing audio: {e}")
            time.sleep(0.1)

    def _get_speech_context(self, min_duration=2.0):
        if not self.is_speaking:
            return None
        target = int(self.sample_rate * min_duration)
        if len(self.continuous_speech_buffer) >= target:
            return np.array(list(self.continuous_speech_buffer)[-target:])
        return None

    async def _synthesize_text(self, text: str):
        try:
            if not self.vv_client:
                return
            query = await self.vv_client.create_audio_query(text, speaker=self.speaker_id)
            wav_data = await query.synthesis(speaker=self.speaker_id)
            self._play_audio(wav_data, self.volume)
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
        # Close vv_client if it supports async closing
        try:
            if self.vv_client:
                close_coro = getattr(self.vv_client, "close", None)
                if callable(close_coro):
                    res = close_coro()
                    if asyncio.iscoroutine(res):
                        await res
        except Exception:
            pass
        finally:
            self.vv_client = None

    def _play_audio(self, wav_data: bytes, volume: float):
        audio_array = np.frombuffer(wav_data, dtype=np.int16)
        adjusted = (audio_array * max(0.0, min(2.0, volume))).astype(np.int16)
        adjusted_bytes = adjusted.tobytes()
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
        try:
            stream.write(adjusted_bytes)
        finally:
            stream.stop_stream()
            stream.close()

    def get_text_history(self):
        return self.text_buffer.copy()

    def clear_text_history(self):
        self.text_buffer.clear()
        self.last_recognized_text = ""


async def run_cli():
    parser = argparse.ArgumentParser(description="Integrated real-time recognition + synthesis")
    parser.add_argument("--model", default="large", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model")
    parser.add_argument("--speaker-id", type=int, default=1431611904, help="AivisSpeech speaker ID")
    parser.add_argument("--language", default="ja", help="Recognition language")
    parser.add_argument("--silence-threshold", type=float, default=0.028, help="RMS threshold for speech")
    parser.add_argument("--no-speak", action="store_true", help="Disable auto synthesis")
    parser.add_argument("--volume", type=float, default=1.0, help="Playback volume (0.0-2.0)")
    parser.add_argument("--endpoint", default="http://localhost:10101", help="AivisSpeech endpoint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="faster-whisper device")
    parser.add_argument("--compute-type", default="int8", help="faster-whisper compute type")
    args = parser.parse_args()

    system = IntegratedSpeechSystem(
        whisper_model=args.model,
        speaker_id=args.speaker_id,
        language=args.language,
        silence_threshold=args.silence_threshold,
        auto_synthesize=(not args.no_speak),
        volume=args.volume,
        device=args.device,
        compute_type=args.compute_type,
    )
    # Bind the loop for cross-thread scheduling
    system.loop = asyncio.get_running_loop()

    if not await system.initialize_aivisspeech(endpoint=args.endpoint):
        return 1
    try:
        system.start_recording()
        while True:
            await asyncio.sleep(0.1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n🛑 Interrupted by user")
    finally:
        system.stop_recording()
        await system.close_async()
    return 0


if __name__ == "__main__":
    asyncio.run(run_cli())
