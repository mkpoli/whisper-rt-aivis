#!/usr/bin/env python3
"""
Quick Start Examples

Ready-to-run examples for immediate testing of the Whisper RT AivisSpeech system.
"""

import asyncio
import sys
import os

# Add the project root to the path so we can import from lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib import WhisperRecognizer, AivisSpeechSynthesizer, IntegratedSpeechSystem


async def quick_recognition():
    """Quick test of speech recognition."""
    print("🎯 Quick Recognition Test (5 seconds)")
    print("=" * 40)

    recognizer = WhisperRecognizer(
        model_name="base",  # Fast model for quick testing
        language="ja",
        silence_threshold=0.028,
    )

    try:
        recognizer.start_recording()
        print("💬 Speak now for 5 seconds...")
        await asyncio.sleep(5)
    finally:
        recognizer.stop_recording()

    print("✅ Recognition test completed!")


async def quick_synthesis():
    """Quick test of text-to-speech."""
    print("\n🎤 Quick Synthesis Test")
    print("=" * 40)

    try:
        async with AivisSpeechSynthesizer(volume=0.8) as tts:
            print("🔊 Synthesizing Japanese...")
            await tts.synthesize_once("こんにちは、テストです")
            await asyncio.sleep(1)

            print("🔊 Synthesizing English...")
            await tts.synthesize_once("Hello, this is a test")
            await asyncio.sleep(1)

        print("✅ Synthesis test completed!")
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        print("💡 Make sure AivisSpeech is running on localhost:10101")


async def quick_integrated():
    """Quick test of the integrated system."""
    print("\n🚀 Quick Integrated System Test (10 seconds)")
    print("=" * 50)

    system = IntegratedSpeechSystem(
        whisper_model="base", language="ja", auto_synthesize=True, volume=0.7
    )

    # Initialize AivisSpeech
    if not await system.initialize_aivisspeech():
        print("❌ Failed to initialize AivisSpeech")
        print("💡 Make sure AivisSpeech is running on localhost:10101")
        return

    try:
        system.start_recording()
        print("💬 Speak now for 10 seconds (will auto-synthesize)...")
        await asyncio.sleep(10)
    finally:
        system.stop_recording()
        await system.close_async()

    print("✅ Integrated system test completed!")


async def main():
    """Run quick start examples."""
    print("🚀 Whisper RT AivisSpeech - Quick Start")
    print("=" * 60)
    print("This will test all three components quickly.")
    print()

    # Run examples
    await quick_recognition()
    await quick_synthesis()
    await quick_integrated()

    print("\n🎉 All quick start tests completed!")
    print("\n💡 Next steps:")
    print("   • Use 'python cli/recognizer_cli.py' for recognition only")
    print("   • Use 'python cli/synthesizer_cli.py' for synthesis only")
    print("   • Use 'python cli/integrated_cli.py' for the full system")


if __name__ == "__main__":
    asyncio.run(main())
