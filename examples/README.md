# Examples

This directory contains examples demonstrating how to use the Whisper RT AivisSpeech library.

## Quick Start

**Immediate Testing** (Recommended first):
```bash
# Windows (PowerShell)
uv sync && .venv\Scripts\activate
whisper-recognize --model base --language ja

# macOS/Linux
uv sync && source .venv/bin/activate
whisper-recognize --model base --language ja
```

This will test speech recognition with the base model.

## Available Examples

### 1. Quick Start (`quick_start.py`)
- **Purpose**: Test all components immediately
- **Duration**: ~30 seconds total
- **Use case**: First-time setup verification
- **Run**: `python examples/quick_start.py`

### 2. GPU Usage (`gpu_usage.py`) 
- **Purpose**: Demonstrate CUDA acceleration
- **Duration**: ~45 seconds total  
- **Use case**: Performance optimization
- **Run**: `python examples/gpu_usage.py`
- **Requirements**: CUDA + PyTorch with CUDA support

### 3. Audio Calibration (`calibration.py`)
- **Purpose**: Optimize VAD parameters for your environment
- **Duration**: Variable (interactive)
- **Use case**: Fine-tuning for specific environments
- **Run**: `python examples/calibration.py`

### 4. Basic Usage (`basic_usage.py`)
- **Purpose**: Comprehensive demonstration of all features
- **Duration**: ~45 seconds total
- **Use case**: Learning the full API
- **Run**: `python examples/basic_usage.py`

## Direct Usage Examples

### Speech Recognition Only
```bash
# Quick test with CLI (recommended)
# Windows (PowerShell)
uv sync && .venv\Scripts\activate
whisper-recognize --model base --language ja

# macOS/Linux
uv sync && source .venv/bin/activate
whisper-recognize --model base --language ja

# Or test examples
python examples/quick_start.py
```

### Text-to-Speech Only
```bash
# Quick test with CLI
# Windows (PowerShell)
uv sync && .venv\Scripts\activate
whisper-synthesize --text "Hello, world!"

# macOS/Linux
uv sync && source .venv/bin/activate
whisper-synthesize --text "Hello, world!"

# Or test examples
python examples/quick_start.py
```

### Integrated System
```bash
# Quick test with CLI
# Windows (PowerShell)
uv sync && .venv\Scripts\activate
whisper-integrated --whisper-model base --auto-synthesize

# macOS/Linux
uv sync && source .venv/bin/activate
whisper-integrated --whisper-model base --auto-synthesize

# Or test examples
python examples/quick_start.py
```

## GPU Acceleration

For best performance with CUDA:
```bash
# GPU recognition
whisper-recognize --device cuda --compute-type float16

# GPU integrated system
whisper-integrated --device cuda --compute-type float16
```

## Library Usage

### Basic Recognition
```python
from lib import WhisperRecognizer

recognizer = WhisperRecognizer(
    model_name="base",
    language="ja",
    silence_threshold=0.028
)

recognizer.start_recording()
# ... do something ...
recognizer.stop_recording()

history = recognizer.get_text_history()
```

### Basic Synthesis
```python
from lib import AivisSpeechSynthesizer

async with AivisSpeechSynthesizer(volume=0.8) as tts:
    await tts.synthesize_once("Hello, world!")
```

### Integrated System
```python
from lib import IntegratedSpeechSystem

system = IntegratedSpeechSystem(
    whisper_model="base",
    language="ja",
    auto_synthesize=True
)

await system.initialize_aivisspeech()
system.start_recording()
# ... do something ...
system.stop_recording()
await system.close_async()
```

## CLI Tools

The library provides command-line tools that work after activating the virtual environment:

- `whisper-recognize` - Speech recognition only
- `whisper-synthesize` - Text-to-speech only  
- `whisper-integrated` - Complete system

## Recommended Workflow

1. **Setup**: 
   - Windows (PowerShell): `uv sync && .venv\Scripts\activate`
   - macOS/Linux: `uv sync && source .venv/bin/activate`
2. **Test Recognition**: `whisper-recognize --model base`
3. **Test Synthesis**: `whisper-synthesize --text "Test"`
4. **Test Integrated**: `whisper-integrated --auto-synthesize`
5. **GPU Usage**: `whisper-recognize --device cuda --compute-type float16`

The CLI tools are the primary way to use the system. The Python examples are for learning the API and debugging.
