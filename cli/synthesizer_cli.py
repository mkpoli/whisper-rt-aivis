#!/usr/bin/env python3
"""
AivisSpeech Synthesizer CLI

Command-line interface for text-to-speech synthesis.
"""

import asyncio
import argparse
import sys
import warnings

from lib.synthesizer.synthesizer import AivisSpeechSynthesizer
from lib.config_kdl import load_kdl_config, apply_config_over_args

# Suppress noisy warnings by default
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")


async def main():
    parser = argparse.ArgumentParser(description="Real-time AivisSpeech TTS")
    parser.add_argument(
        "--config", type=str, help="Path to KDL config file (section: synthesizer)"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:10101", help="AivisSpeech HTTP endpoint"
    )
    parser.add_argument(
        "--speaker-id", type=int, default=888753760, help="AivisSpeech speaker ID"
    )
    parser.add_argument(
        "--volume", type=float, default=1.0, help="Playback volume (0.0 - 2.0)"
    )
    parser.add_argument("--text", type=str, help="Synthesize a single line of text")
    parser.add_argument(
        "--stdin", action="store_true", help="Read lines from stdin and synthesize each"
    )
    args = parser.parse_args()

    # Build presence map for precedence
    presence = {}
    for action in parser._actions:  # type: ignore[attr-defined]
        if not getattr(action, "option_strings", None):
            continue
        dest = getattr(action, "dest", None)
        if not dest:
            continue
        presence[dest] = any(opt in sys.argv for opt in action.option_strings)

    if getattr(args, "config", None):
        try:
            cfg = load_kdl_config(args.config, section="synthesizer")
            args = apply_config_over_args(args, cfg, flag_presence_lookup=presence)
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")

    if not args.text and not args.stdin:
        print("❌ Provide --text or --stdin")
        return 1

    async with AivisSpeechSynthesizer(
        endpoint=args.endpoint, speaker_id=args.speaker_id, volume=args.volume
    ) as tts:
        if args.text:
            await tts.synthesize_once(args.text)
        if args.stdin:
            print("✍️  Type text lines to synthesize (Ctrl+C to exit)\n")
            try:
                for line in sys.stdin:
                    line = line.strip()
                    if not line:
                        continue
                    await tts.synthesize_once(line)
            except KeyboardInterrupt:
                print("\n🛑 Stopped")
    return 0


def main_sync():
    """Synchronous wrapper for the CLI entry point."""
    return asyncio.run(main())


if __name__ == "__main__":
    main_sync()
