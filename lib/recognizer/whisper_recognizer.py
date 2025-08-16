#!/usr/bin/env python3
"""
Whisper Recognizer Library

Core speech recognition class using OpenAI Whisper.
"""

import time
import threading
from collections import deque
from typing import Optional, Callable, List
import os
import sys
from pathlib import Path

import numpy as np
import pyaudio
from faster_whisper import WhisperModel


class WhisperRecognizer:
    """Real-time speech recognition using OpenAI Whisper."""
    
    def __init__(self, 
                 model_name: str = "large",
                 language: str = "ja",
                 silence_threshold: float = 0.028,
                 sample_rate: int = 16000,
                 chunk_duration: float = 2.0,
                 device: str = "cpu",
                 compute_type: str = "int8",
                 on_text_callback: Optional[Callable[[str, float, float], None]] = None):
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
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 10))
        self.continuous_speech_buffer = deque(maxlen=int(self.sample_rate * 15))
        
        # Speech detection
        self.is_speaking = False
        self.speech_start_time = 0
        self.continuous_speech_duration = 0
        self.last_speech_time = 0
        
        # Processing
        self.processing_thread = None
        self.last_recognized_text = ""
        self.text_buffer = []
        
        # Initialize Whisper
        print(f"🎯 Loading Whisper model: {model_name} [device={device}, compute={compute_type}]")
        if self.device == "cuda" and sys.platform == "win32":
            self._prepare_cuda_dlls()
        self.whisper = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("✅ Whisper model loaded successfully!")

    def _prepare_cuda_dlls(self) -> None:
        """Prepare CUDA DLLs for Windows."""
        try:
            import torch  # noqa: F401
            torch_lib = Path(__import__("torch").__file__).parent / "lib"
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
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
        self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.processing_thread.start()
        print("🚀 Whisper Recognizer Started!")
        print(f"🎯 Model: {self.model_name}")
        print(f"🌍 Language: {self.language}")
        print(f"🔊 Silence Threshold: {self.silence_threshold}")
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
            self._detect_speech(audio_data)
        return (in_data, pyaudio.paContinue)

    def _detect_speech(self, audio_data):
        """Detect speech based on audio levels."""
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
        """Main audio processing loop."""
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
                                
                                # Call callback if provided
                                if self.on_text_callback:
                                    self.on_text_callback(text, current_rms, self.continuous_speech_duration)
                                    
                        except Exception as e:
                            print(f"❌ Error processing audio: {e}")
            
            time.sleep(0.1)

    def _get_speech_context(self, min_duration: float = 2.0):
        """Get audio context for speech processing."""
        if not self.is_speaking:
            return None
        target = int(self.sample_rate * min_duration)
        if len(self.continuous_speech_buffer) >= target:
            return np.array(list(self.continuous_speech_buffer)[-target:])
        return None

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
