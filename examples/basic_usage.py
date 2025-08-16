#!/usr/bin/env python3
"""
Basic Usage Example

Demonstrates how to use the Whisper RT AivisSpeech library.
"""

import asyncio
from lib import WhisperRecognizer, AivisSpeechSynthesizer, IntegratedSpeechSystem


async def recognizer_example():
    """Example of using just the recognizer."""
    print("🎯 Recognizer Example")
    print("=" * 50)
    
    recognizer = WhisperRecognizer(
        model_name="base",
        language="ja",
        silence_threshold=0.028
    )
    
    try:
        recognizer.start_recording()
        print("💬 Speak for 10 seconds...")
        await asyncio.sleep(10)
    finally:
        recognizer.stop_recording()
    
    # Show results
    history = recognizer.get_text_history()
    print(f"\n📝 Recognized text: {history}")


async def synthesizer_example():
    """Example of using just the synthesizer."""
    print("\n🎤 Synthesizer Example")
    print("=" * 50)
    
    async with AivisSpeechSynthesizer(volume=0.8) as tts:
        await tts.synthesize_once("こんにちは、世界！")
        await asyncio.sleep(1)
        await tts.synthesize_once("Hello, world!")


async def integrated_example():
    """Example of using the integrated system."""
    print("\n🚀 Integrated System Example")
    print("=" * 50)
    
    system = IntegratedSpeechSystem(
        whisper_model="base",
        language="ja",
        auto_synthesize=True,
        volume=0.7
    )
    
    # Initialize AivisSpeech
    if not await system.initialize_aivisspeech():
        print("❌ Failed to initialize AivisSpeech")
        return
    
    try:
        system.start_recording()
        print("💬 Speak for 15 seconds (will auto-synthesize)...")
        await asyncio.sleep(15)
    finally:
        system.stop_recording()
        await system.close_async()
    
    # Show results
    history = system.get_text_history()
    print(f"\n📝 Processed text: {history}")


async def main():
    """Run all examples."""
    print("🎯 Whisper RT AivisSpeech Library Examples")
    print("=" * 60)
    
    # Run examples
    await recognizer_example()
    await synthesizer_example()
    await integrated_example()
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
