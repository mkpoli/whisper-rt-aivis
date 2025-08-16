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

### 1. Real-time Whisper Recognition

Run the standalone real-time recognizer (tools with clear hierarchy):

```bash
python realtime_whisper.py
```

**Command line options**:
- `--model`: Whisper model size (tiny, base, small, medium, large) - default: base
- `--chunk-duration`: Audio chunk duration in seconds - default: 2.0
- `--sample-rate`: Audio sample rate - default: 16000
- `--language`: Language code (ja, en, zh, etc.) - default: ja
- `--silence-threshold`: Minimum audio level for speech detection (0.0-1.0) - default: 0.01
- `--min-speech-duration`: Minimum speech duration to process (seconds) - default: 0.5
- `--list-devices`: List available audio input devices

**Quick Start Options**:
```bash
# High accuracy (best quality, slower)
python realtime_whisper.py --model large --silence-threshold 0.028

# Balanced (good quality, moderate speed)
python realtime_whisper.py --model medium --silence-threshold 0.028

# Fast (lower quality, fastest speed)
python realtime_whisper.py --model tiny --silence-threshold 0.028
```

**GPU Usage (CUDA)**:
```bash
# VAD mode (recommended):
uv run python realtime_whisper.py --model large --device cuda --compute-type float16 --mode vad --silence-threshold 0.028 --concise-logs --show-setup

# RAW mode (no gating):
uv run python realtime_whisper.py --model large --device cuda --compute-type float16 --mode raw --concise-logs --show-setup
```
Use `--fw-vad` to enable internal faster-whisper VAD and `--dedup-seconds 3` to suppress repeated lines.

**Examples**:
```bash
# Use tiny model for faster processing
python realtime_whisper.py --model tiny

# Use English language
python realtime_whisper.py --language en

# Use 1-second chunks for more responsive recognition
python realtime_whisper.py --chunk-duration 1.0

# Adjust silence threshold to reduce noise
python realtime_whisper.py --silence-threshold 0.02

# Use longer speech duration requirement
python realtime_whisper.py --min-speech-duration 1.0

# List audio devices
python realtime_whisper.py --list-devices
```

### 2. Audio Level Calibration

Before using the real-time recognizer, calibrate your silence threshold:

```bash
# Monitor audio levels to find optimal threshold
uv run python audio_level_monitor.py
```

This tool shows real-time audio levels and recommends a silence threshold based on your environment.

### 3. Pre-configured Scripts & Structure

**High Accuracy Test** (Recommended for best results):
```bash
uv run python test_realtime.py
```
Uses the `large` model with optimized settings for your environment.

**Interactive Model Selection**:
```bash
uv run python accuracy_vs_speed.py
```
Choose between different accuracy vs speed trade-offs interactively.

**Project Layout**:
```
tools/
  recognition/        # real-time whisper tools
  synthesis/          # AivisSpeech TTS tools
  integrated/         # combined speech-to-speech

realtime_whisper.py           # CLI entry (recognition)
realtime_synthesis.py         # CLI entry (synthesis)
integrated_speech_system.py   # CLI entry (integrated)
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

