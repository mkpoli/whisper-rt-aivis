# Examples

This directory contains examples demonstrating how to use the Whisper RT AivisSpeech library.

## Basic Usage

Run the basic examples:

```bash
python examples/basic_usage.py
```

This will demonstrate:
- Speech recognition only
- Text-to-speech synthesis only  
- Integrated recognition + synthesis

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

The library also provides command-line tools:

- `whisper-recognize` - Speech recognition only
- `whisper-synthesize` - Text-to-speech only  
- `whisper-integrated` - Complete system

See the main README for CLI usage details.
