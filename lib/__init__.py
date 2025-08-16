"""
Whisper RT AivisSpeech Library

A library for near real-time speech recognition and synthesis.
"""

from .recognizer.whisper_recognizer import WhisperRecognizer
from .synthesizer.synthesizer import AivisSpeechSynthesizer
from .integrated_system import IntegratedSpeechSystem

__version__ = "0.1.0"

__all__ = [
    "WhisperRecognizer",
    "AivisSpeechSynthesizer", 
    "IntegratedSpeechSystem"
]
