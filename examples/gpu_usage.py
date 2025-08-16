#!/usr/bin/env python3
"""
GPU Usage Examples

Examples demonstrating GPU acceleration with CUDA for optimal performance.
"""

import asyncio
import sys
import os

# Add the project root to the path so we can import from lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib import WhisperRecognizer, IntegratedSpeechSystem


async def gpu_recognition_example():
    """Example of GPU-accelerated speech recognition."""
    print("🚀 GPU-Accelerated Recognition Example")
    print("=" * 50)
    print("Using CUDA with float16 for optimal performance")
    print()

    recognizer = WhisperRecognizer(
        model_name="large",  # Best accuracy model
        language="ja",
        silence_threshold=0.028,
        device="cuda",
        compute_type="float16",  # Best performance on modern GPUs
    )

    try:
        recognizer.start_recording()
        print("💬 Speak now for 10 seconds (GPU accelerated)...")
        await asyncio.sleep(10)
    finally:
        recognizer.stop_recording()

    print("✅ GPU recognition test completed!")


async def gpu_integrated_example():
    """Example of GPU-accelerated integrated system."""
    print("\n🚀 GPU-Accelerated Integrated System Example")
    print("=" * 55)
    print("Combines GPU recognition with synthesis for best performance")
    print()

    system = IntegratedSpeechSystem(
        whisper_model="large",
        language="ja",
        auto_synthesize=True,
        volume=0.7,
        device="cuda",
        compute_type="float16",
    )

    # Initialize AivisSpeech
    if not await system.initialize_aivisspeech():
        print("❌ Failed to initialize AivisSpeech")
        print("💡 Make sure AivisSpeech is running on localhost:10101")
        return

    try:
        system.start_recording()
        print("💬 Speak now for 15 seconds (GPU + synthesis)...")
        await asyncio.sleep(15)
    finally:
        system.stop_recording()
        await system.close_async()

    print("✅ GPU integrated system test completed!")


async def performance_comparison():
    """Compare CPU vs GPU performance."""
    print("\n⚡ Performance Comparison Example")
    print("=" * 45)
    print("Testing CPU vs GPU with same model")
    print()

    # Test CPU
    print("🖥️  Testing CPU performance...")
    cpu_recognizer = WhisperRecognizer(
        model_name="base",
        language="ja",
        silence_threshold=0.028,
        device="cpu",
        compute_type="int8",
    )

    try:
        cpu_recognizer.start_recording()
        print("💬 Speak for 5 seconds (CPU)...")
        await asyncio.sleep(5)
    finally:
        cpu_recognizer.stop_recording()

    # Test GPU
    print("\n🚀 Testing GPU performance...")
    gpu_recognizer = WhisperRecognizer(
        model_name="base",
        language="ja",
        silence_threshold=0.028,
        device="cuda",
        compute_type="float16",
    )

    try:
        gpu_recognizer.start_recording()
        print("💬 Speak for 5 seconds (GPU)...")
        await asyncio.sleep(5)
    finally:
        gpu_recognizer.stop_recording()

    print("\n✅ Performance comparison completed!")
    print("💡 GPU should be noticeably faster for the same model")


async def main():
    """Run GPU usage examples."""
    print("🚀 Whisper RT AivisSpeech - GPU Usage Examples")
    print("=" * 60)
    print("These examples demonstrate GPU acceleration with CUDA.")
    print("Make sure you have CUDA and PyTorch installed.")
    print()

    try:
        # Run examples
        await gpu_recognition_example()
        await gpu_integrated_example()
        await performance_comparison()

        print("\n🎉 All GPU examples completed!")
        print("\n💡 For CLI usage with GPU:")
        print("   • python cli/recognizer_cli.py --device cuda --compute-type float16")
        print("   • python cli/integrated_cli.py --device cuda --compute-type float16")

    except Exception as e:
        print(f"\n❌ GPU test failed: {e}")
        print("💡 Make sure:")
        print("   • CUDA is installed and working")
        print("   • PyTorch with CUDA support is installed")
        print("   • GPU drivers are up to date")


if __name__ == "__main__":
    asyncio.run(main())
