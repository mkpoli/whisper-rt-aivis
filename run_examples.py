#!/usr/bin/env python3
"""
Example Runner Script

Quick way to run examples from the project root.
"""

import sys
import os
import subprocess


def main():
    """Run examples based on command line arguments."""
    if len(sys.argv) < 2:
        print("🎯 Whisper RT AivisSpeech - Example Runner")
        print("=" * 50)
        print("Usage:")
        print("  python run_examples.py quick     # Quick start (recommended first)")
        print("  python run_examples.py gpu       # GPU usage examples")
        print("  python run_examples.py calibrate # Audio calibration")
        print("  python run_examples.py basic     # Basic usage examples")
        print("  python run_examples.py all       # Run all examples")
        print()
        print("Examples:")
        print("  python run_examples.py quick")
        print("  python run_examples.py gpu")
        print()
        print("💡 For production use, prefer CLI tools:")
        print("  # Windows (PowerShell):")
        print("  uv sync && .venv\\Scripts\\activate")
        print("  whisper-recognize --model base")
        print("  whisper-synthesize --text 'Hello'")
        print("  whisper-integrated --auto-synthesize")
        print("  # macOS/Linux:")
        print("  uv sync && source .venv/bin/activate")
        return

    example = sys.argv[1].lower()

    examples_dir = "examples"
    if not os.path.exists(examples_dir):
        print(f"❌ Examples directory not found: {examples_dir}")
        return

    if example == "quick":
        script = "quick_start.py"
        print("🚀 Running Quick Start Example...")
    elif example == "gpu":
        script = "gpu_usage.py"
        print("🚀 Running GPU Usage Examples...")
    elif example == "calibrate":
        script = "calibration.py"
        print("🎵 Running Audio Calibration Examples...")
    elif example == "basic":
        script = "basic_usage.py"
        print("📚 Running Basic Usage Examples...")
    elif example == "all":
        print("🎯 Running All Examples...")
        scripts = ["quick_start.py", "gpu_usage.py", "calibration.py", "basic_usage.py"]
        for script in scripts:
            script_path = os.path.join(examples_dir, script)
            if os.path.exists(script_path):
                print(f"\n🔄 Running {script}...")
                try:
                    subprocess.run(
                        [sys.executable, script_path], cwd=examples_dir, check=True
                    )
                except subprocess.CalledProcessError:
                    print(f"❌ {script} failed")
                except KeyboardInterrupt:
                    print(f"⏹️  {script} interrupted by user")
                    break
        print("\n✅ All examples completed!")
        return
    else:
        print(f"❌ Unknown example: {example}")
        print("Available: quick, gpu, calibrate, basic, all")
        return

    script_path = os.path.join(examples_dir, script)
    if not os.path.exists(script_path):
        print(f"❌ Example script not found: {script_path}")
        return

    print(f"📁 Running: {script}")
    print(f"💡 Press Ctrl+C to stop any example")
    print()

    try:
        subprocess.run([sys.executable, script], cwd=examples_dir, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Example failed")
    except KeyboardInterrupt:
        print(f"\n⏹️  Example interrupted by user")


if __name__ == "__main__":
    main()
