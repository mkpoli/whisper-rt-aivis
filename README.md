# Whisper RT AivisSpeech

A real-time speech-to-speech processing system that combines OpenAI Whisper for speech recognition with AivisSpeech for voice synthesis, creating a complete audio processing pipeline.

## What This System Does

This system captures live audio input, converts it to text using Whisper in real-time, and then synthesizes the text back to speech using AivisSpeech. It's designed for applications where you need to process speech through an AI pipeline - such as real-time translation, speech enhancement, content moderation, or creating speech processing workflows.

The system maintains low latency by processing audio in configurable chunks while providing real-time feedback and control over the recognition and synthesis parameters.

## Use Cases

This system is designed for applications that require real-time speech processing:

- **Real-time Translation**: Convert speech from one language to another with minimal delay
- **Speech Enhancement**: Process and improve speech quality in real-time
- **Content Moderation**: Monitor and filter speech content as it's spoken
- **Accessibility Tools**: Provide real-time speech-to-text and text-to-speech conversion
- **Speech Analysis**: Real-time transcription and analysis of live audio
- **AI Speech Workflows**: Create custom speech processing pipelines for research or applications
- **Live Broadcasting**: Process speech in real-time for streaming or broadcasting applications

## Features

- **Real-time Speech Recognition**: Continuous microphone input processing with OpenAI Whisper
- **Sound Level Detection**: Prevents hallucination by only processing audio above threshold
- **Voice Synthesis**: Integration with AivisSpeech for text-to-speech
- **Volume Control**: Adjustable audio playback volume
- **Multi-language Support**: Support for Japanese, English, and other languages
- **Configurable Models**: Choose from different Whisper model sizes (tiny, base, small, medium, large)
- **Audio Level Monitoring**: Real-time visualization and calibration tools
- **GPU Acceleration**: CUDA support via PyTorch+cu129 (uv-configured)
- **Low Latency Processing**: Optimized for real-time applications with configurable chunk sizes

## Project Structure

```
whisper-rt-aivis/
├── lib/                          # Core library
│   ├── recognizer/              # Speech recognition components
│   │   ├── __init__.py
│   │   └── whisper_recognizer.py
│   ├── synthesizer/             # Text-to-speech components
│   │   ├── __init__.py
│   │   └── synthesizer.py
│   ├── __init__.py
│   └── integrated_system.py     # Combined system
├── cli/                         # Command-line tools
│   ├── __init__.py
│   ├── recognizer_cli.py        # Recognition CLI
│   ├── synthesizer_cli.py       # Synthesis CLI
│   └── integrated_cli.py        # Integrated system CLI
├── examples/                    # Usage examples
│   ├── README.md
│   └── basic_usage.py
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd whisper-rt-aivis
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Install additional system dependencies** (if needed):
   - **Windows**: PyAudio should work out of the box
   - **macOS**: `brew install portaudio`
   - **Linux**: `sudo apt-get install portaudio19-dev python3-pyaudio`

## Usage

### 1. Quick Start

**Setup and activate virtual environment:**
```bash
# Windows (PowerShell)
uv sync && .venv\Scripts\activate

# macOS/Linux  
uv sync && source .venv/bin/activate
```

**Test the system:**
```bash
# Test speech recognition
whisper-recognize --model base --language ja

# Test text-to-speech
whisper-synthesize --text "Hello, world!"

# Test integrated system
whisper-integrated --whisper-model base --auto-synthesize
```

### 2. Command Line Tools

The project provides three main CLI tools (after `uv venv activate`):

#### Speech Recognition Only
```bash
# Basic recognition
whisper-recognize

# With custom settings
whisper-recognize --model large --language en --silence-threshold 0.02

# GPU acceleration
whisper-recognize --device cuda --compute-type float16
```

#### Text-to-Speech Only
```bash
# Single text synthesis
whisper-synthesize --text "こんにちは、世界！"

# Interactive mode
whisper-synthesize --stdin
```

#### Audio Level Monitoring
```bash
# Monitor for 30 seconds
whisper-audio-monitor --duration 30

# Monitor indefinitely
whisper-audio-monitor --duration 0
```

#### Integrated System (Recognition + Synthesis)
```bash
# Complete system
whisper-integrated

# Custom configuration
whisper-integrated --model medium --language ja --volume 0.8
```

### 2. Library Usage

#### Basic Recognition
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

#### Basic Synthesis
```python
from lib import AivisSpeechSynthesizer

async with AivisSpeechSynthesizer(volume=0.8) as tts:
    await tts.synthesize_once("Hello, world!")
```

#### Integrated System
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

### 3. Examples

Run the provided examples:
```bash
python examples/basic_usage.py
```

### 4. Jupyter Notebook

```python
# Run the simple real-time recognizer
simple_realtime_whisper()

# Or use the full-featured version
from realtime_whisper import RealtimeWhisperRecognizer

with RealtimeWhisperRecognizer(model_name="base", language="ja") as recognizer:
    recognizer.start_recording()
    # Recognition runs in background
    # Press Ctrl+C to stop
```

### 5. Volume Control in Audio Playback

The `play_wav` function now supports volume adjustment:

```python
# 50% volume (quieter)
play_wav(wav_data, volume=0.5)

# Normal volume
play_wav(wav_data, volume=1.0)

# 150% volume (louder)
play_wav(wav_data, volume=1.5)
```

## How It Works

### Complete Speech-to-Speech Pipeline

This system creates a real-time audio processing pipeline that:

1. **Audio Capture**: Continuously captures audio from microphone in configurable chunks
2. **Sound Level Detection**: Monitors RMS (Root Mean Square) audio levels to detect speech
3. **Speech Validation**: Only processes audio chunks above the silence threshold
4. **Buffer Management**: Maintains rolling audio buffers for seamless processing
5. **Whisper Processing**: Sends validated audio chunks to Whisper model for transcription
6. **Text Processing**: The recognized text can be processed, modified, or enhanced as needed
7. **AivisSpeech Synthesis**: Converts the processed text back to speech using AivisSpeech
8. **Audio Output**: Plays the synthesized speech with configurable volume control

### Audio Processing Flow

```
Microphone → Audio Buffer → Sound Level Detection → Speech Validation → Whisper Model → Text Output
     ↓           ↓              ↓                    ↓                ↓           ↓
  16-bit    Rolling        RMS Level           Threshold Check   Transcription   Display
  PCM      Buffer         Calculation         (Above Level?)    (Real-time)     (Timestamped)
                                    ↓
                              Text Processing → AivisSpeech Synthesis → Audio Playback
                                    ↓                ↓              ↓
                              (Optional)         Voice Output    Volume Control
                              Modifications     (Real-time)     (Configurable)
```

### Real-time Recognition Process

1. **Audio Capture**: Continuously captures audio from microphone in configurable chunks
2. **Sound Level Detection**: Monitors RMS (Root Mean Square) audio levels to detect speech
3. **Speech Validation**: Only processes audio chunks above the silence threshold
4. **Buffer Management**: Maintains rolling audio buffers for seamless processing
5. **Whisper Processing**: Sends validated audio chunks to Whisper model for transcription
6. **Real-time Output**: Displays recognized text with timestamps and audio levels

## Configuration

### Whisper Models

- **tiny**: Fastest, least accurate (39M parameters)
- **base**: Good balance of speed/accuracy (74M parameters)
- **small**: Better accuracy, slower (244M parameters)
- **medium**: High accuracy, slower (769M parameters)
- **large**: Best accuracy, slowest (1550M parameters)

### Audio Settings

- **Sample Rate**: 16000 Hz (optimal for Whisper)
- **Chunk Duration**: 2.0 seconds (configurable)
- **Buffer Size**: 10 seconds (rolling buffer)
- **Format**: 16-bit PCM, mono channel

## Troubleshooting

### Common Issues

1. **No audio input detected**:
   - Check microphone permissions
   - Use `--list-devices` to verify audio device
   - Ensure microphone is not muted

2. **High CPU usage**:
   - Use smaller Whisper model (tiny/base)
   - Increase chunk duration
   - Check for background processes

3. **Audio quality issues**:
   - Ensure quiet environment
   - Check microphone quality
   - Adjust chunk duration

4. **Too many false positives (hallucination)**:
   - Increase silence threshold: `--silence-threshold 0.02`
   - Increase minimum speech duration: `--min-speech-duration 1.0`
   - Use audio level monitor to calibrate threshold

5. **Missing speech recognition**:
   - Decrease silence threshold: `--silence-threshold 0.005`
   - Decrease minimum speech duration: `--min-speech-duration 0.3`
   - Check microphone sensitivity

### Performance Tips

- Use `tiny` or `base` models for real-time applications
- Keep chunk duration between 1-3 seconds
- Close unnecessary applications to free up CPU
- Use SSD storage for faster model loading

## Dependencies

- **Core**: Python 3.13+
- **Audio**: PyAudio, numpy
- **ML**: OpenAI Whisper, torch
- **Speech**: SpeechRecognition
- **HTTP**: aiohttp, requests
- **Voice**: voicevox-client

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

