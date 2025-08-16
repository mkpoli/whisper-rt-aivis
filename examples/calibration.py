#!/usr/bin/env python3
"""
Audio Calibration Examples

Examples for calibrating audio levels and Voice Activity Detection (VAD) parameters.
"""

import asyncio
import sys
import os

# Add the project root to the path so we can import from lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib import WhisperRecognizer
from cli.audio_level_monitor import AudioLevelMonitor


async def audio_level_calibration():
    """Calibrate audio levels for optimal VAD."""
    print("🎵 Audio Level Calibration")
    print("=" * 40)
    print("This will help you find the optimal silence threshold.")
    print()

    monitor = AudioLevelMonitor()

    try:
        print("🔊 Starting audio level monitoring...")
        print("💬 Speak at different volumes and watch the levels")
        print("📊 Look for the typical level when you're speaking vs. silent")
        print("⏹️  Press Ctrl+C to stop monitoring")
        print()

        monitor.start_monitoring()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    finally:
        monitor.stop_monitoring()

    print("\n💡 Calibration tips:")
    print("   • Normal speech: typically 0.02-0.15")
    print("   • Background noise: typically 0.005-0.02")
    print("   • Recommended threshold: 2-3x background noise")
    print("   • Try 0.028 for most environments")


async def vad_parameter_testing():
    """Test different VAD parameters to find optimal settings."""
    print("\n🎯 VAD Parameter Testing")
    print("=" * 40)
    print("Testing different silence thresholds and speech durations")
    print()

    # Test different thresholds
    thresholds = [0.01, 0.02, 0.028, 0.05, 0.1]

    for threshold in thresholds:
        print(f"🔊 Testing threshold: {threshold}")
        print(f"💬 Speak for 5 seconds with threshold {threshold}")

        recognizer = WhisperRecognizer(
            model_name="base",
            language="ja",
            silence_threshold=threshold,
            chunk_duration=2.0,
        )

        try:
            recognizer.start_recording()
            await asyncio.sleep(5)
        finally:
            recognizer.stop_recording()

        print(f"✅ Threshold {threshold} test completed")
        print()

    print("💡 VAD parameter recommendations:")
    print("   • Low threshold (0.01-0.02): Sensitive, may pick up background noise")
    print("   • Medium threshold (0.028-0.05): Balanced, good for most use cases")
    print("   • High threshold (0.1+): Less sensitive, only picks up loud speech")


async def environment_adaptation():
    """Test how the system adapts to different environments."""
    print("\n🏠 Environment Adaptation Testing")
    print("=" * 45)
    print("Testing dynamic silence threshold adaptation")
    print()

    # Test with dynamic silence enabled
    print("🔄 Testing with dynamic silence threshold...")
    recognizer = WhisperRecognizer(
        model_name="base", language="ja", silence_threshold=0.028, chunk_duration=2.0
    )

    try:
        recognizer.start_recording()
        print("💬 Speak with natural pauses for 10 seconds...")
        print("   (The system should adapt to your speaking pattern)")
        await asyncio.sleep(10)
    finally:
        recognizer.stop_recording()

    print("✅ Dynamic adaptation test completed!")
    print("\n💡 Dynamic silence benefits:")
    print("   • Adapts to your speaking speed")
    print("   • Prevents cutting off sentences")
    print("   • Better for natural conversation")


async def main():
    """Run calibration examples."""
    print("🎵 Whisper RT AivisSpeech - Audio Calibration Examples")
    print("=" * 65)
    print("These examples help you optimize audio parameters for your environment.")
    print()

    try:
        # Run examples
        await audio_level_calibration()
        await vad_parameter_testing()
        await environment_adaptation()

        print("\n🎉 All calibration examples completed!")
        print("\n💡 Next steps:")
        print("   • Use the optimal threshold in your CLI commands")
        print("   • Adjust min_speech_duration if needed")
        print("   • Enable dynamic_silence for natural speech")

    except Exception as e:
        print(f"\n❌ Calibration failed: {e}")
        print("💡 Make sure your microphone is working and accessible")


if __name__ == "__main__":
    asyncio.run(main())
