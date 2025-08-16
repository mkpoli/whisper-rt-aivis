#!/usr/bin/env python3
"""
Simple test script for real-time Whisper recognition
"""

from realtime_whisper import RealtimeWhisperRecognizer
import time

def test_realtime_recognition():
    """Test the real-time recognition with the best accuracy model"""
    print("🚀 Starting high-accuracy real-time Whisper recognition test...")
    print("This will use the 'large' model for maximum accuracy.")
    print("✨ Features: Dynamic silence detection, continuous speech handling")
    print("🎯 Optimized for: Long sentences, natural pauses, reduced hallucination")
    print("Speak into your microphone and press Ctrl+C to stop.\n")
    
    try:
        with RealtimeWhisperRecognizer(
            model_name="large",  # Use large model for best accuracy
            chunk_duration=2.0,  # 2.0 second chunks for better context
            language="ja",       # Japanese language
            silence_threshold=0.028,  # Optimized threshold from your audio analysis
            min_speech_duration=1.0,  # Longer speech duration for better accuracy
            dynamic_silence=True      # Enable dynamic silence threshold
        ) as recognizer:
            recognizer.start_recording()
            
            # Keep running until interrupted
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    test_realtime_recognition()
