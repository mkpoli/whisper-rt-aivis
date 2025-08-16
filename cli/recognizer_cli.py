#!/usr/bin/env python3
"""
Whisper Recognizer CLI

Command-line interface for speech recognition using Whisper.
"""

import asyncio
import argparse
import signal
import sys

from lib.recognizer.whisper_recognizer import WhisperRecognizer


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\n🛑 Interrupted by user")
    sys.exit(0)


async def main():
    parser = argparse.ArgumentParser(description="Real-time Whisper Speech Recognition")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], 
                       help="Whisper model size")
    parser.add_argument("--language", default="ja", help="Recognition language code")
    parser.add_argument("--silence-threshold", type=float, default=0.028, 
                       help="RMS threshold for speech detection (0.0-1.0)")
    parser.add_argument("--chunk-duration", type=float, default=2.0, 
                       help="Audio chunk duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], 
                       help="Device for Whisper processing")
    parser.add_argument("--compute-type", default="int8", 
                       help="Compute type for faster-whisper")
    args = parser.parse_args()

    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)

    # Create recognizer
    recognizer = WhisperRecognizer(
        model_name=args.model,
        language=args.language,
        silence_threshold=args.silence_threshold,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        device=args.device,
        compute_type=args.compute_type
    )

    try:
        recognizer.start_recording()
        print("💬 Speak now! Press Ctrl+C to stop.\n")
        
        # Keep the main thread alive
        while True:
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        recognizer.stop_recording()
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())
