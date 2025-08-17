#!/usr/bin/env python3
"""
Audio Level Monitor CLI

Command-line interface for monitoring audio levels and calibrating silence thresholds.
"""

import asyncio
import argparse
import signal
import sys
import warnings

from .audio_level_monitor import AudioLevelMonitor

# Suppress noisy warnings by default
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\n🛑 Interrupted by user")
    sys.exit(0)


async def main():
    parser = argparse.ArgumentParser(description="Audio Level Monitor for Calibration")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Monitoring duration in seconds (0 for infinite)")
    parser.add_argument("--sample-rate", type=int, default=16000, 
                       help="Audio sample rate")
    parser.add_argument("--chunk-size", type=int, default=1024, 
                       help="Audio chunk size")
    args = parser.parse_args()

    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)

    monitor = AudioLevelMonitor(
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size
    )

    try:
        if args.duration > 0:
            print(f"🔊 Monitoring audio levels for {args.duration} seconds...")
            print("💡 Speak normally to see your speech levels")
            print("💡 Stay quiet to see your silence levels")
            print("💡 Press Ctrl+C to stop early\n")
            
            await asyncio.sleep(args.duration)
        else:
            print("🔊 Monitoring audio levels indefinitely...")
            print("💡 Speak normally to see your speech levels")
            print("💡 Stay quiet to see your silence levels")
            print("💡 Press Ctrl+C to stop\n")
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        monitor.stop()
        print("\n📊 Final Statistics:")
        monitor.print_statistics()


def main_sync():
    """Synchronous wrapper for the CLI entry point."""
    return asyncio.run(main())


if __name__ == "__main__":
    main_sync()
