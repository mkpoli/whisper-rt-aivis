#!/usr/bin/env python3
"""
Integrated Speech System CLI

Command-line interface for the complete speech recognition + synthesis system.
"""

import asyncio
import argparse
import sys
import warnings

from lib.integrated_system import IntegratedSpeechSystem
from lib.config_kdl import load_kdl_config, apply_config_over_args

# Suppress noisy warnings by default
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")


async def run_cli():
    parser = argparse.ArgumentParser(
        description="Integrated real-time recognition + synthesis"
    )
    parser.add_argument(
        "--config", type=str, help="Path to KDL config file (section: integrated)"
    )
    parser.add_argument(
        "--model",
        default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model",
    )
    parser.add_argument(
        "--speaker-id", type=int, default=888753760, help="AivisSpeech speaker ID"
    )
    parser.add_argument("--language", default="ja", help="Recognition language")
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.028,
        help="RMS threshold for speech detection",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=2.0,
        help="Audio chunk duration in seconds",
    )
    parser.add_argument(
        "--no-speak", action="store_true", help="Disable auto synthesis"
    )
    parser.add_argument(
        "--volume", type=float, default=1.0, help="Playback volume (0.0-2.0)"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:10101", help="AivisSpeech endpoint"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="faster-whisper device"
    )
    parser.add_argument(
        "--compute-type", default="int8", help="faster-whisper compute type"
    )
    args = parser.parse_args()

    # Build presence map: which flags were explicitly provided
    presence = {}
    for action in parser._actions:  # type: ignore[attr-defined]
        if not getattr(action, "option_strings", None):
            continue
        dest = getattr(action, "dest", None)
        if not dest:
            continue
        presence[dest] = any(opt in sys.argv for opt in action.option_strings)

    # Load and apply KDL config if provided
    if getattr(args, "config", None):
        try:
            cfg = load_kdl_config(args.config, section="integrated")
            args = apply_config_over_args(args, cfg, flag_presence_lookup=presence)
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")

    system = IntegratedSpeechSystem(
        whisper_model=args.model,
        speaker_id=args.speaker_id,
        language=args.language,
        silence_threshold=args.silence_threshold,
        chunk_duration=args.chunk_duration,
        auto_synthesize=(not args.no_speak),
        volume=args.volume,
        device=args.device,
        compute_type=args.compute_type,
    )

    # Bind the loop for cross-thread scheduling
    system.loop = asyncio.get_running_loop()

    if not await system.initialize_aivisspeech(endpoint=args.endpoint):
        return 1

    try:
        system.start_recording()
        while True:
            await asyncio.sleep(0.1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n🛑 Interrupted by user")
    finally:
        system.stop_recording()
        await system.close_async()

    return 0


def main():
    """Synchronous wrapper for the CLI entry point."""
    return asyncio.run(run_cli())


if __name__ == "__main__":
    main()
