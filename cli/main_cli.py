#!/usr/bin/env python3
"""
Main CLI Entry Point

Provides an interactive interface for configuring and running CLI tools
when no specific options are specified.
"""

import argparse
import sys
import asyncio
from typing import Optional, Dict, Any, List, Tuple
import json
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

from .integrated_cli import run_cli as run_integrated
from .recognizer_cli import main as run_recognizer
from .synthesizer_cli import main as run_synthesizer
from .audio_monitor_cli import main_sync as run_audio_monitor


def print_banner():
    """Print a welcome banner for the interactive CLI."""
    print("🎤 Whisper RT AivisSpeech CLI")
    print("=" * 40)


def get_user_choice() -> str:
    """Get user choice for which CLI tool to run."""
    print("\nAvailable tools:")
    print("1. 🎯 Integrated System (Recognition + Synthesis)")
    print("2. 🎧 Speech Recognition Only")
    print("3. 🔊 Text-to-Speech Only")
    print("4. 📊 Audio Level Monitor")
    print("5. ❌ Exit")

    while True:
        try:
            choice = input("\nSelect an option (1-5): ").strip()
            if choice in ["1", "2", "3", "4", "5"]:
                return choice
            else:
                print("❌ Please enter a number between 1 and 5")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def get_whisper_model() -> str:
    """Get user choice for Whisper model."""
    print("\n🎯 Whisper Model Selection:")
    print("1. tiny   - Fastest, least accurate")
    print("2. base   - Fast, good accuracy")
    print("3. small  - Balanced speed/accuracy")
    print("4. medium - Slower, better accuracy")
    print("5. large  - Slowest, best accuracy")

    while True:
        try:
            choice = input("\nSelect model (1-5) [default: 3]: ").strip()
            if not choice:
                return "small"
            if choice in ["1", "2", "3", "4", "5"]:
                models = ["tiny", "base", "small", "medium", "large"]
                return models[int(choice) - 1]
            else:
                print("❌ Please enter a number between 1 and 5")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def get_language() -> str:
    """Get user choice for language."""
    print("\n🌍 Language Selection:")
    print("1. ja (Japanese)")
    print("2. en (English)")
    print("3. zh (Chinese)")
    print("4. ko (Korean)")
    print("5. Other (custom)")

    while True:
        try:
            choice = input("\nSelect language (1-5) [default: 1]: ").strip()
            if not choice:
                return "ja"
            if choice == "1":
                return "ja"
            elif choice == "2":
                return "en"
            elif choice == "3":
                return "zh"
            elif choice == "4":
                return "ko"
            elif choice == "5":
                custom = input("Enter language code (e.g., fr, de, es): ").strip()
                if custom:
                    return custom
                else:
                    print("❌ Language code cannot be empty")
            else:
                print("❌ Please enter a number between 1 and 5")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def get_device() -> str:
    """Get user choice for device."""
    print("\n💻 Device Selection:")
    print("1. cpu - CPU processing (slower)")
    print("2. cuda - GPU acceleration (faster)")

    while True:
        try:
            choice = input("\nSelect device (1-2) [default: 1]: ").strip()
            if not choice:
                return "cpu"
            if choice == "1":
                return "cpu"
            elif choice == "2":
                return "cuda"
            else:
                print("❌ Please enter 1 or 2")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def get_compute_type() -> str:
    """Get user choice for compute type."""
    print("\n⚡ Compute Type Selection:")
    print("1. int8 - Fastest, least accurate")
    print("2. float16 - Balanced speed/accuracy")
    print("3. float32 - Slowest, most accurate")

    while True:
        try:
            choice = input("\nSelect compute type (1-3) [default: 1]: ").strip()
            if not choice:
                return "int8"
            if choice == "1":
                return "int8"
            elif choice == "2":
                return "float16"
            elif choice == "3":
                return "float32"
            else:
                print("❌ Please enter a number between 1 and 3")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def get_silence_threshold() -> float:
    """Get user choice for silence threshold."""
    print("\n🔇 Silence Threshold:")
    print("Lower values = more sensitive to quiet speech")
    print("Higher values = less sensitive, fewer false positives")

    while True:
        try:
            threshold = input("\nEnter threshold (0.01-0.1) [default: 0.028]: ").strip()
            if not threshold:
                return 0.028
            threshold_float = float(threshold)
            if 0.01 <= threshold_float <= 0.1:
                return threshold_float
            else:
                print("❌ Please enter a value between 0.01 and 0.1")
        except ValueError:
            print("❌ Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def get_chunk_duration() -> float:
    """Get user choice for chunk duration."""
    print("\n⏱️  Chunk Duration:")
    print("Shorter chunks = lower latency, more processing")
    print("Longer chunks = higher latency, less processing")

    while True:
        try:
            duration = input(
                "\nEnter duration in seconds (1.0-5.0) [default: 2.0]: "
            ).strip()
            if not duration:
                return 2.0
            duration_float = float(duration)
            if 1.0 <= duration_float <= 5.0:
                return duration_float
            else:
                print("❌ Please enter a value between 1.0 and 5.0")
        except ValueError:
            print("❌ Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def get_volume() -> float:
    """Get user choice for volume."""
    print("\n🔊 Volume Control:")
    print("0.0 = silent, 1.0 = normal, 2.0 = double volume")

    while True:
        try:
            volume = input("\nEnter volume (0.0-2.0) [default: 1.0]: ").strip()
            if not volume:
                return 1.0
            volume_float = float(volume)
            if 0.0 <= volume_float <= 2.0:
                return volume_float
            else:
                print("❌ Please enter a value between 0.0 and 2.0")
        except ValueError:
            print("❌ Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def _fetch_speakers(endpoint: str) -> Any:
    """Fetch speakers JSON from AivisSpeech `/speakers` endpoint."""
    url = endpoint.rstrip("/") + "/speakers"
    with urlrequest.urlopen(url, timeout=10) as resp:  # nosec - local endpoint by user
        charset = resp.headers.get_content_charset() or "utf-8"
        data = resp.read().decode(charset, errors="replace")
        return json.loads(data)


def _build_speaker_options(speakers_json: Any) -> List[Tuple[int, str]]:
    """Flatten speakers JSON into a list of (id, label). Supports VoiceVox-compatible schema."""
    options: List[Tuple[int, str]] = []
    if isinstance(speakers_json, list):
        for speaker in speakers_json:
            try:
                base_name = (
                    speaker.get("name") or speaker.get("speakerName") or "Unknown"
                )
                styles = speaker.get("styles")
                if isinstance(styles, list) and styles:
                    for style in styles:
                        style_id = (
                            style.get("id")
                            or style.get("speaker_id")
                            or style.get("styleId")
                        )
                        style_name = style.get("name") or style.get("style") or ""
                        if isinstance(style_id, int):
                            label = (
                                f"{base_name}{' - ' + style_name if style_name else ''}"
                            )
                            options.append((style_id, label))
                else:
                    # Some engines may expose a flat list with ids
                    style_id = speaker.get("id") or speaker.get("speaker_id")
                    if isinstance(style_id, int):
                        options.append((style_id, base_name))
            except Exception:
                # Skip malformed entries
                continue
    return options


def get_speaker_id(endpoint: str) -> int:
    """Fetch available speakers from endpoint and let the user choose one.

    Exits with error if fetch fails or no speakers are found.
    """
    print("\n👤 Speaker Selection:")
    try:
        speakers_json = _fetch_speakers(endpoint)
        options = _build_speaker_options(speakers_json)
        if not options:
            print(
                "❌ No speakers discovered at the endpoint. Ensure the engine is running and has speakers."
            )
            sys.exit(1)

        # Sort options by label for consistent display
        options.sort(key=lambda x: x[1].lower())

        print("Discovered speakers:")
        for idx, (sid, label) in enumerate(options, start=1):
            print(f"{idx}. {label} (id: {sid})")

        while True:
            try:
                choice = input(
                    "\nSelect a speaker (1-{} ) [default: 1]: ".format(len(options))
                ).strip()
                if not choice:
                    return options[0][0]
                choice_int = int(choice)
                if 1 <= choice_int <= len(options):
                    return options[choice_int - 1][0]
                else:
                    print("❌ Please enter a valid number in range")
            except ValueError:
                print("❌ Please enter a valid number")
    except (URLError, HTTPError, TimeoutError, ValueError) as e:
        print(f"❌ Failed to fetch speakers from endpoint: {e}")
        sys.exit(1)
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
        sys.exit(0)


def get_endpoint() -> str:
    """Get user choice for AivisSpeech endpoint."""
    print("\n🌐 AivisSpeech Endpoint:")
    print("Default: http://localhost:10101")

    while True:
        try:
            endpoint = input(
                "\nEnter endpoint [default: http://localhost:10101]: "
            ).strip()
            if not endpoint:
                return "http://localhost:10101"
            if endpoint.startswith(("http://", "https://")):
                return endpoint
            else:
                print("❌ Please enter a valid URL starting with http:// or https://")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def configure_common_options() -> Dict[str, Any]:
    """Configure common options for all tools."""
    config: Dict[str, Any] = {}

    print("\n🔧 Common Configuration:")
    config["model"] = get_whisper_model()
    config["language"] = get_language()
    config["device"] = get_device()
    config["compute_type"] = get_compute_type()
    # Store numeric values as strings to avoid overly narrow dict type inference
    config["silence_threshold"] = str(get_silence_threshold())
    config["chunk_duration"] = str(get_chunk_duration())

    return config


def configure_synthesis_options() -> Dict[str, Any]:
    """Configure synthesis-specific options."""
    config: Dict[str, Any] = {}

    print("\n🔊 Synthesis Configuration:")
    # Endpoint must be known before selecting speaker
    config["endpoint"] = get_endpoint()
    # Store as strings for consistent argv usage
    config["speaker_id"] = str(get_speaker_id(config["endpoint"]))
    config["volume"] = str(get_volume())

    return config


def configure_integrated_options() -> Dict[str, Any]:
    """Configure integrated system options."""
    config: Dict[str, Any] = configure_common_options()
    synthesis_config: Dict[str, Any] = configure_synthesis_options()
    config.update(synthesis_config)

    print("\n🎯 Integrated System Configuration:")
    print("Auto-synthesis enabled by default")
    config["auto_synthesize"] = True

    return config


def run_recognizer_interactive():
    """Run recognizer with interactive configuration."""
    print("\n🎧 Configuring Speech Recognition...")
    config = configure_common_options()

    print(f"\n🚀 Starting Speech Recognition with:")
    print(f"   Model: {config['model']}")
    print(f"   Language: {config['language']}")
    print(f"   Device: {config['device']}")
    print(f"   Compute Type: {config['compute_type']}")
    print(f"   Silence Threshold: {config['silence_threshold']}")
    print(f"   Chunk Duration: {config['chunk_duration']}")

    # Set up sys.argv to pass configuration to recognizer
    original_argv = sys.argv.copy()
    sys.argv = [
        "whisper-recognize",
        "--model",
        config["model"],
        "--language",
        config["language"],
        "--device",
        config["device"],
        "--compute-type",
        config["compute_type"],
        "--silence-threshold",
        str(config["silence_threshold"]),
        "--chunk-duration",
        str(config["chunk_duration"]),
    ]

    try:
        result = run_recognizer()
        if result != 0:
            print(f"\n⚠️  Recognition exited with code {result}")
    except KeyboardInterrupt:
        print("\n🛑 Recognition stopped")
    finally:
        sys.argv = original_argv


def run_synthesizer_interactive():
    """Run synthesizer with interactive configuration."""
    print("\n🔊 Configuring Text-to-Speech...")
    config = configure_synthesis_options()

    print(f"\n🚀 Starting Text-to-Speech with:")
    print(f"   Speaker ID: {config['speaker_id']}")
    print(f"   Volume: {config['volume']}")
    print(f"   Endpoint: {config['endpoint']}")

    # Ask for text input method
    print("\n📝 Text Input Method:")
    print("1. Single text line")
    print("2. Interactive input (stdin)")

    while True:
        try:
            choice = input("\nSelect method (1-2): ").strip()
            if choice == "1":
                text = input("Enter text to synthesize: ").strip()
                if text:
                    # Set up sys.argv to pass configuration to synthesizer
                    original_argv = sys.argv.copy()
                    sys.argv = [
                        "whisper-synthesize",
                        "--text",
                        text,
                        "--speaker-id",
                        str(config["speaker_id"]),
                        "--volume",
                        str(config["volume"]),
                        "--endpoint",
                        config["endpoint"],
                    ]

                    try:
                        result = asyncio.run(run_synthesizer())
                        if result != 0:
                            print(f"\n⚠️  Synthesis exited with code {result}")
                    except KeyboardInterrupt:
                        print("\n🛑 Synthesis stopped")
                    finally:
                        sys.argv = original_argv
                else:
                    print("❌ Text cannot be empty")
                break
            elif choice == "2":
                # Set up sys.argv to pass configuration to synthesizer
                original_argv = sys.argv.copy()
                sys.argv = [
                    "whisper-synthesize",
                    "--stdin",
                    "--speaker-id",
                    str(config["speaker_id"]),
                    "--volume",
                    str(config["volume"]),
                    "--endpoint",
                    config["endpoint"],
                ]

                try:
                    result = asyncio.run(run_synthesizer())
                    if result != 0:
                        print(f"\n⚠️  Synthesis exited with code {result}")
                except KeyboardInterrupt:
                    print("\n🛑 Synthesis stopped")
                finally:
                    sys.argv = original_argv
                break
            else:
                print("❌ Please enter 1 or 2")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)


def run_integrated_interactive():
    """Run integrated system with interactive configuration."""
    print("\n🎯 Configuring Integrated System...")
    config = configure_integrated_options()

    print(f"\n🚀 Starting Integrated System with:")
    print(f"   Model: {config['model']}")
    print(f"   Language: {config['language']}")
    print(f"   Device: {config['device']}")
    print(f"   Compute Type: {config['compute_type']}")
    print(f"   Silence Threshold: {config['silence_threshold']}")
    print(f"   Chunk Duration: {config['chunk_duration']}")
    print(f"   Speaker ID: {config['speaker_id']}")
    print(f"   Volume: {config['volume']}")
    print(f"   Endpoint: {config['endpoint']}")
    print(f"   Auto-synthesis: {config['auto_synthesize']}")

    # Set up sys.argv to pass configuration to integrated system
    original_argv = sys.argv.copy()
    sys.argv = [
        "whisper-integrated",
        "--model",
        config["model"],
        "--language",
        config["language"],
        "--device",
        config["device"],
        "--compute-type",
        config["compute_type"],
        "--silence-threshold",
        str(config["silence_threshold"]),
        "--chunk-duration",
        str(config["chunk_duration"]),
        "--speaker-id",
        str(config["speaker_id"]),
        "--volume",
        str(config["volume"]),
        "--endpoint",
        config["endpoint"],
    ]

    try:
        result = asyncio.run(run_integrated())
        if result != 0:
            print(f"\n⚠️  Integrated system exited with code {result}")
    except KeyboardInterrupt:
        print("\n🛑 Integrated system stopped")
    finally:
        sys.argv = original_argv


def run_audio_monitor_interactive():
    """Run audio monitor with interactive configuration."""
    print("\n📊 Configuring Audio Level Monitor...")

    print("\n⏱️  Monitoring Duration:")
    print("1. 30 seconds (recommended for calibration)")
    print("2. 60 seconds")
    print("3. Infinite (until interrupted)")

    while True:
        try:
            choice = input("\nSelect duration (1-3) [default: 1]: ").strip()
            if not choice:
                duration = 30
                break
            if choice == "1":
                duration = 30
                break
            elif choice == "2":
                duration = 60
                break
            elif choice == "3":
                duration = 0
                break
            else:
                print("❌ Please enter a number between 1 and 3")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            sys.exit(0)

    print(
        f"\n🚀 Starting Audio Level Monitor for {duration if duration > 0 else 'infinite'} seconds..."
    )

    # Set up sys.argv to pass configuration to audio monitor
    original_argv = sys.argv.copy()
    sys.argv = ["whisper-audio-monitor", "--duration", str(duration)]

    try:
        result = run_audio_monitor()
        if result != 0:
            print(f"\n⚠️  Audio monitor exited with code {result}")
    except KeyboardInterrupt:
        print("\n🛑 Audio monitor stopped")
    finally:
        sys.argv = original_argv


def run_interactive_mode():
    """Run the interactive CLI mode."""
    print_banner()

    while True:
        choice = get_user_choice()

        if choice == "1":
            run_integrated_interactive()
        elif choice == "2":
            run_recognizer_interactive()
        elif choice == "3":
            run_synthesizer_interactive()
        elif choice == "4":
            run_audio_monitor_interactive()
        elif choice == "5":
            print("\n👋 Goodbye!")
            break


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Whisper RT AivisSpeech - Main CLI",
        add_help=False,  # We'll handle help manually
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode (default when no other options specified)",
    )
    parser.add_argument(
        "--integrated", action="store_true", help="Run integrated system directly"
    )
    parser.add_argument(
        "--recognize", action="store_true", help="Run speech recognition directly"
    )
    parser.add_argument(
        "--synthesize", action="store_true", help="Run text-to-speech directly"
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Run audio level monitor directly"
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message"
    )

    # Parse known args to avoid conflicts with other CLIs
    args, unknown = parser.parse_known_args()

    # If help is requested, show help
    if args.help:
        parser.print_help()
        print("\nExamples:")
        print("  whisper-aivis                    # Interactive configuration mode")
        print("  whisper-aivis -i                 # Interactive configuration mode")
        print("  whisper-aivis --integrated      # Run integrated system with defaults")
        print("  whisper-aivis --recognize       # Run recognition with defaults")
        print("  whisper-aivis --synthesize      # Run synthesis with defaults")
        print("  whisper-aivis --monitor         # Run audio monitor with defaults")
        print("\nInteractive mode allows you to configure:")
        print("  • Whisper model (tiny/base/small/medium/large)")
        print("  • Language (ja/en/zh/ko/custom)")
        print("  • Device (cpu/cuda)")
        print("  • Compute type (int8/float16/float32)")
        print("  • Silence threshold (0.01-0.1)")
        print("  • Chunk duration (1.0-5.0 seconds)")
        print("  • Volume (0.0-2.0)")
        print("  • Speaker ID and endpoint")
        return 0

    # If specific tool is requested, run it directly
    if args.integrated:
        print("🚀 Starting Integrated System...")
        return asyncio.run(run_integrated())
    elif args.recognize:
        print("🎧 Starting Speech Recognition...")
        return run_recognizer()
    elif args.synthesize:
        print("🔊 Starting Text-to-Speech...")
        return asyncio.run(run_synthesizer())
    elif args.monitor:
        print("📊 Starting Audio Level Monitor...")
        return run_audio_monitor()

    # If no specific tool is requested, run interactive mode
    if args.interactive or not unknown:
        run_interactive_mode()
        return 0

    # If unknown args are provided, show help
    print("❌ Unknown arguments:", unknown)
    print("Use --help for usage information")
    return 1


if __name__ == "__main__":
    main()
