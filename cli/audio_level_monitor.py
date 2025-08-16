#!/usr/bin/env python3
"""
Audio Level Monitor

This script helps you calibrate the silence threshold for the real-time Whisper recognizer
by showing real-time audio levels from your microphone.
"""

import numpy as np
import pyaudio
import time
from collections import deque

class AudioLevelMonitor:
    def __init__(self, sample_rate=16000, chunk_duration=0.1):
        """
        Initialize audio level monitor
        
        Args:
            sample_rate: Audio sample rate
            chunk_duration: Duration of audio chunks in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_monitoring = False
        
        # Level tracking
        self.levels = deque(maxlen=100)  # Keep last 100 levels
        self.peak_level = 0.0
        
    def start_monitoring(self):
        """Start monitoring audio levels"""
        if self.is_monitoring:
            print("Already monitoring!")
            return
            
        self.is_monitoring = True
        
        # Start audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("🎤 Audio level monitoring started! Press Ctrl+C to stop.")
        print("Speak normally to see your typical speech levels.")
        print("The recommended silence threshold is 2-3x your background noise level.\n")
        
        try:
            while self.is_monitoring:
                # Read audio data
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Calculate RMS level
                audio_float = audio_array.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_float ** 2))
                
                # Track levels
                self.levels.append(rms)
                if rms > self.peak_level:
                    self.peak_level = rms
                
                # Display level with visual indicator
                self._display_level(rms)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        finally:
            self.stop_monitoring()
    
    def _display_level(self, rms):
        """Display audio level with visual indicator"""
        # Create visual bar (50 characters wide)
        bar_width = 50
        level_bar = int(rms * bar_width * 100)  # Scale up for visibility
        
        # Clamp bar length
        level_bar = min(level_bar, bar_width)
        
        # Create visual representation
        bar = "█" * level_bar + "░" * (bar_width - level_bar)
        
        # Color coding based on level
        if rms < 0.01:
            level_icon = "🔇"
            level_desc = "Silence"
        elif rms < 0.05:
            level_icon = "🔈"
            level_desc = "Whisper"
        elif rms < 0.15:
            level_icon = "🔉"
            level_desc = "Normal Speech"
        elif rms < 0.3:
            level_icon = "🔊"
            level_desc = "Loud Speech"
        else:
            level_icon = "🔊"
            level_desc = "Very Loud"
        
        # Display with timestamp
        timestamp = time.strftime("%H:%M:%S")
        print(f"\r[{timestamp}] {level_icon} {rms:.4f} {bar} {level_desc:<12}", end="", flush=True)
    
    def stop_monitoring(self):
        """Stop monitoring audio levels"""
        self.is_monitoring = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        self.audio.terminate()
        
        # Print summary
        if self.levels:
            avg_level = np.mean(self.levels)
            min_level = np.min(self.levels)
            max_level = np.max(self.levels)
            
            print(f"\n\n📊 Audio Level Summary:")
            print(f"   Average: {avg_level:.4f}")
            print(f"   Minimum: {min_level:.4f}")
            print(f"   Maximum: {max_level:.4f}")
            print(f"   Peak:    {self.peak_level:.4f}")
            print(f"\n💡 Recommended silence threshold: {max(0.01, avg_level * 2):.4f}")
            print(f"   (2x average level, minimum 0.01)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()

def main():
    """Run the audio level monitor"""
    print("🎵 Audio Level Monitor for Whisper Recognition")
    print("=" * 50)
    
    try:
        with AudioLevelMonitor() as monitor:
            monitor.start_monitoring()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
