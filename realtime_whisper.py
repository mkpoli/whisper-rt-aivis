#!/usr/bin/env python3
"""
Real-time Whisper Speech Recognition

This script provides real-time speech recognition using OpenAI's Whisper model.
It continuously listens to microphone input and transcribes speech in real-time.
"""

import numpy as np
import pyaudio
import threading
import queue
import time
from collections import deque
from faster_whisper import WhisperModel
import argparse
import sys
import os
from pathlib import Path
import warnings

class RealtimeWhisperRecognizer:
    def __init__(self, model_name="base", chunk_duration=2.0, sample_rate=16000, language="ja", 
                 silence_threshold=0.01, min_speech_duration=0.5, dynamic_silence=True,
                 mode: str = "vad", device: str = "cpu", compute_type: str = "int8",
                 fw_vad: bool = False, condition_on_previous_text: bool = False,
                 beam_size: int = 5, best_of: int = 5, temperature: float = 0.0,
                 length_penalty: float = 1.0, dedup_seconds: float = 2.0,
                 concise_logs: bool = False, show_setup: bool = False):
        """
        Initialize real-time Whisper recognizer
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
            chunk_duration: Duration of audio chunks in seconds
            sample_rate: Audio sample rate
            language: Language code for recognition (e.g., "ja", "en", "zh")
            silence_threshold: Minimum audio level to consider as speech (0.0 to 1.0)
            min_speech_duration: Minimum duration of speech to process (seconds)
        """
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.language = language
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.dynamic_silence = dynamic_silence
        # Recognition mode: "vad" (level/duration gated) or "raw" (always process)
        self.mode = mode  # "vad" | "raw"
        # faster-whisper decode options
        self.fw_vad = fw_vad
        self.condition_on_previous_text = condition_on_previous_text
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.dedup_seconds = dedup_seconds
        self.concise_logs = concise_logs
        self._last_text = ""
        self._last_text_ts = 0.0
        self.chunk_size = int(sample_rate * chunk_duration)
        self.show_setup = show_setup
        self._setup_messages: list[str] = []
        
        # Load Whisper model
        self.device = device
        self.compute_type = compute_type
        self._setup_messages.append(
            f"Loading Whisper model: {model_name} [device={self.device}, compute={self.compute_type}]"
        )
        # On Windows, help the dynamic loader find cuDNN from PyTorch if CUDA is requested
        if self.device == "cuda" and sys.platform == "win32":
            self._prepare_cuda_dlls()
        self.model = WhisperModel(model_name, device=self.device, compute_type=self.compute_type)
        self._setup_messages.append("Model loaded successfully!")
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=int(sample_rate * 10))  # 10 second buffer
        
        # Speech detection
        self.speech_buffer = deque(maxlen=int(sample_rate * 5))  # 5 second speech buffer
        self.last_speech_time = 0
        self.is_speaking = False
        self.speech_start_time = 0
        self.continuous_speech_duration = 0
        self.silence_duration = 0
        self.speech_history = deque(maxlen=10)  # Keep track of recent speech segments
        
        # Continuous speech buffer for better context
        self.continuous_speech_buffer = deque(maxlen=int(sample_rate * 15))  # 15 second buffer
        
        # Threading
        self.processing_thread = None
        
    def start_recording(self):
        """Start real-time recording and recognition"""
        if self.is_recording:
            print("Already recording!")
            return
            
        self.is_recording = True
        
        # Start audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.processing_thread.start()
        
        print("\n====================")
        print("🎤 Real-time Whisper")
        print("====================")
        print(f"Language: {self.language}")
        print(f"Chunk duration: {self.chunk_duration}s")
        print(f"Sample rate: {self.sample_rate}Hz")
        if self.mode == "vad":
            print(f"Silence threshold: {self.silence_threshold}")
            print(f"Min speech duration: {self.min_speech_duration}s")
            print("🔊 VAD mode: level/duration gating enabled")
        else:
            print("🎧 RAW mode: processing every chunk without level/duration gating")
        print(f"Device: {self.device} | Compute: {self.compute_type}")
        print(f"fw_vad={self.fw_vad} | cond_prev={self.condition_on_previous_text} | beam={self.beam_size} | temp={self.temperature}")
        if self.concise_logs:
            print("Log mode: concise (text only)")
        else:
            print("Log mode: verbose (levels, durations)")
        print("Press Ctrl+C to stop.\n")
        if self.show_setup and self._setup_messages:
            print("-- Setup -----------------")
            for line in self._setup_messages:
                print(line)
            print("--------------------------\n")

    def _prepare_cuda_dlls(self) -> None:
        try:
            import torch  # noqa: F401
            torch_lib = Path(__import__("torch").__file__).parent / "lib"
            if torch_lib.exists():
                # Add PyTorch's DLL folder so cudnn_ops64_9.dll can be found
                os.add_dll_directory(str(torch_lib))
                self._setup_messages.append(f"✅ Added CUDA DLL path: {torch_lib}")
        except Exception as e:
            self._setup_messages.append(f"⚠️ CUDA DLL prep failed (will try anyway): {e}")

    def stop_recording(self):
        """Stop recording and recognition"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()
        
        self.audio.terminate()
        print("🛑 Recognition stopped.")

    def _get_speech_context(self, min_duration=2.0):
        """Get the best audio context for speech processing"""
        if not self.is_speaking:
            return None
            
        # Calculate how much audio we need
        target_samples = int(self.sample_rate * min_duration)
        
        # Get audio from continuous buffer
        if len(self.continuous_speech_buffer) >= target_samples:
            # Get the most recent audio that covers the speech duration
            audio_data = list(self.continuous_speech_buffer)[-target_samples:]
            return np.array(audio_data)
        
        return None
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input"""
        if self.is_recording:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            # Add to buffers
            self.audio_buffer.extend(audio_data)
            self.continuous_speech_buffer.extend(audio_data)
            
            # Check audio level for speech detection (VAD mode only)
            if self.mode == "vad":
                self._detect_speech(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _detect_speech(self, audio_data):
        """Detect if audio contains speech based on level and duration"""
        # Calculate RMS (Root Mean Square) of audio data
        audio_float = audio_data.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Add to speech buffer
        self.speech_buffer.append(rms)
        
        current_time = time.time()
        
        # Check if current audio level is above threshold
        if rms > self.silence_threshold:
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = current_time
                self.continuous_speech_duration = 0
                self.silence_duration = 0
            else:
                # Continue speaking
                self.continuous_speech_duration = current_time - self.speech_start_time
                self.silence_duration = 0
            self.last_speech_time = current_time
        else:
            # Silent period
            if self.is_speaking:
                self.silence_duration = current_time - self.last_speech_time
                
                # Only end speech if we've been silent for a longer period
                # This prevents cutting off sentences with natural pauses
                if self.dynamic_silence:
                    silence_threshold = max(1.0, self.continuous_speech_duration * 0.3)  # Dynamic threshold
                else:
                    silence_threshold = 0.5  # Fixed threshold
                
                if self.silence_duration > silence_threshold:
                    # Speech ended
                    self.is_speaking = False
                    # Store speech duration for analysis
                    if self.continuous_speech_duration > 0:
                        self.speech_history.append(self.continuous_speech_duration)
    
    def _process_audio(self):
        """Process audio chunks in real-time"""
        while self.is_recording:
            if len(self.audio_buffer) >= self.chunk_size:
                # Get chunk of audio
                chunk = np.array(list(self.audio_buffer)[:self.chunk_size])
                
                # Remove processed audio from buffer
                for _ in range(self.chunk_size):
                    if self.audio_buffer:
                        self.audio_buffer.popleft()
                
                if self.mode == "raw":
                    # RAW: process current chunk directly
                    try:
                        audio_float = chunk.astype(np.float32) / 32768.0
                        segments, info = self.model.transcribe(
                            audio_float,
                            language=self.language,
                            vad_filter=self.fw_vad,
                            condition_on_previous_text=self.condition_on_previous_text,
                            beam_size=self.beam_size,
                            best_of=self.best_of,
                            temperature=self.temperature,
                            length_penalty=self.length_penalty,
                        )
                        text = " ".join([segment.text for segment in segments]).strip()
                        if text and not self._is_duplicate(text):
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"[{timestamp}] 🎯 {text}")
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                else:
                    # VAD: Only process if we detected speech
                    if self.is_speaking:
                        # Get better audio context for processing
                        speech_context = self._get_speech_context(min_duration=2.0)
                        
                        if speech_context is not None:
                            # Calculate current audio level for display
                            audio_float = speech_context.astype(np.float32) / 32768.0
                            current_rms = np.sqrt(np.mean(audio_float ** 2))
                            
                            # Process with Whisper using better context
                            try:
                                # Transcribe with better audio context
                                segments, info = self.model.transcribe(
                                    audio_float,
                                    language=self.language,
                                    vad_filter=self.fw_vad,
                                    condition_on_previous_text=self.condition_on_previous_text,
                                    beam_size=self.beam_size,
                                    best_of=self.best_of,
                                    temperature=self.temperature,
                                    length_penalty=self.length_penalty,
                                )
                                text = " ".join([segment.text for segment in segments]).strip()
                                
                                if text and not self._is_duplicate(text):  # Only print if there's actual text
                                    timestamp = time.strftime("%H:%M:%S")
                                    if self.concise_logs:
                                        print(f"[{timestamp}] 🎯 {text}")
                                    else:
                                        speech_duration = f"{self.continuous_speech_duration:.1f}s"
                                        context_duration = f"{len(speech_context)/self.sample_rate:.1f}s"
                                        print(f"[{timestamp}] 🔊 {current_rms:.4f} ⏱️{speech_duration} 📏{context_duration} 🎯 {text}")
                                else:
                                    # Show audio level even when no text detected
                                    if not self.concise_logs:
                                        timestamp = time.strftime("%H:%M:%S")
                                        speech_duration = f"{self.continuous_speech_duration:.1f}s"
                                        context_duration = f"{len(speech_context)/self.sample_rate:.1f}s"
                                        print(f"[{timestamp}] 🔊 {current_rms:.4f} ⏱️{speech_duration} 📏{context_duration} (no speech detected)")
                                
                            except Exception as e:
                                print(f"Error processing audio: {e}")
                # Silence indicator only for VAD mode
                if self.mode == "vad" and not self.is_speaking and not self.concise_logs:
                    if time.time() % 5 < 0.1:  # Every ~5 seconds
                        timestamp = time.strftime("%H:%M:%S")
                        avg_speech_duration = np.mean(self.speech_history) if self.speech_history else 0
                        print(f"[{timestamp}] 🔇 Silence (waiting for speech) - Avg speech: {avg_speech_duration:.1f}s")
                
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()

    def _is_duplicate(self, text: str) -> bool:
        now = time.time()
        if text == self._last_text and (now - self._last_text_ts) < self.dedup_seconds:
            return True
        self._last_text = text
        self._last_text_ts = now
        return False

def list_audio_devices():
    """List available audio input devices"""
    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"  {i}: {device_info['name']}")
    p.terminate()

def main():
    parser = argparse.ArgumentParser(description="Real-time Whisper Speech Recognition")
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--chunk-duration", type=float, default=2.0,
                       help="Duration of audio chunks in seconds (default: 2.0)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Audio sample rate (default: 16000)")
    parser.add_argument("--language", default="ja",
                       help="Language code for recognition (default: ja)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="faster-whisper device (cpu or cuda)")
    parser.add_argument("--compute-type", default="int8",
                       help="faster-whisper compute type (e.g., int8, int8_float16, float16, float32)")
    parser.add_argument("--fw-vad", action="store_true", help="Enable faster-whisper internal VAD filter")
    parser.add_argument("--condition-prev", action="store_true", help="Condition on previous text (may increase drift)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--best-of", type=int, default=5, help="Number of candidates to sample")
    parser.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty for beam search")
    parser.add_argument("--dedup-seconds", type=float, default=2.0, help="Suppress identical outputs within this window")
    parser.add_argument("--concise-logs", action="store_true", help="Only print recognized text (no levels/durations)")
    parser.add_argument("--quiet-warnings", action="store_true", help="Suppress noisy runtime warnings (ctranslate2)")
    parser.add_argument("--show-setup", action="store_true", help="Show setup details (model load, CUDA DLL path)")
    parser.add_argument("--silence-threshold", type=float, default=0.01,
                       help="Minimum audio level to consider as speech (0.0 to 1.0, default: 0.01)")
    parser.add_argument("--min-speech-duration", type=float, default=0.5,
                       help="Minimum duration of speech to process (seconds, default: 0.5)")
    parser.add_argument("--no-dynamic-silence", action="store_true",
                       help="Disable dynamic silence threshold (use fixed 0.5s)")
    parser.add_argument("--mode", choices=["vad", "raw"], default="vad",
                       help='Recognition mode: "vad" uses level/duration gating; "raw" processes all audio')
    parser.add_argument("--list-devices", action="store_true",
                       help="List available audio input devices")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    if args.quiet_warnings:
        warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

    try:
        with RealtimeWhisperRecognizer(
            model_name=args.model,
            chunk_duration=args.chunk_duration,
            sample_rate=args.sample_rate,
            language=args.language,
            silence_threshold=args.silence_threshold,
            min_speech_duration=args.min_speech_duration,
            dynamic_silence=not args.no_dynamic_silence,
            mode=args.mode,
            device=args.device,
            compute_type=args.compute_type,
            fw_vad=args.fw_vad,
            condition_on_previous_text=args.condition_prev,
            beam_size=args.beam_size,
            best_of=args.best_of,
            temperature=args.temperature,
            length_penalty=args.length_penalty,
            dedup_seconds=args.dedup_seconds,
            concise_logs=args.concise_logs,
            show_setup=args.show_setup,
        ) as recognizer:
            recognizer.start_recording()
            
            # Keep running until interrupted
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
