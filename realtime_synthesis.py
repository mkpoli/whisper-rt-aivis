#!/usr/bin/env python3
"""
Real-time Synthesis (AivisSpeech)

This script performs text-to-speech using AivisSpeech with convenient real-time modes:
- Single-shot: pass --text "..."
- Continuous: pass --stdin and type lines to synthesize
"""

import asyncio
import argparse
import sys
from typing import Optional

import numpy as np
import pyaudio
import vvclient


class AivisSpeechSynthesizer:
	def __init__(self, endpoint: str = "http://localhost:10101", speaker_id: int = 1431611904, volume: float = 1.0):
		self.endpoint = endpoint
		self.speaker_id = speaker_id
		self.volume = max(0.0, min(2.0, volume))
		self._client: Optional[vvclient.Client] = None
		self._pa: Optional[pyaudio.PyAudio] = None

	async def __aenter__(self):
		self._client = vvclient.Client(self.endpoint)
		# Touch the engine to verify connectivity
		version = await self._client.fetch_engine_version()
		print(f"🎤 AivisSpeech connected: {version}")
		self._pa = pyaudio.PyAudio()
		return self

	async def __aexit__(self, exc_type, exc, tb):
		if self._pa:
			self._pa.terminate()
			self._pa = None
		# vvclient.Client supports async context normally; close session if available
		try:
			if self._client:
				await self._client.close()
		except Exception:
			pass
		self._client = None

	async def synthesize_once(self, text: str):
		if not text or not text.strip():
			return
		assert self._client is not None, "Client not initialized"
		# Create query and synthesize
		query = await self._client.create_audio_query(text, speaker=self.speaker_id)
		wav_data = await query.synthesis(speaker=self.speaker_id)
		self._play_wav(wav_data, self.volume)

	def _play_wav(self, wav_data: bytes, volume: float):
		assert self._pa is not None, "PyAudio not initialized"
		audio_array = np.frombuffer(wav_data, dtype=np.int16)
		adjusted = (audio_array * volume).astype(np.int16)
		adjusted_bytes = adjusted.tobytes()
		stream = self._pa.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
		try:
			stream.write(adjusted_bytes)
		finally:
			stream.stop_stream()
			stream.close()


async def main():
	parser = argparse.ArgumentParser(description="Real-time AivisSpeech TTS")
	parser.add_argument("--endpoint", default="http://localhost:10101", help="AivisSpeech HTTP endpoint")
	parser.add_argument("--speaker-id", type=int, default=1431611904, help="AivisSpeech speaker ID")
	parser.add_argument("--volume", type=float, default=1.0, help="Playback volume (0.0 - 2.0)")
	parser.add_argument("--text", type=str, help="Synthesize a single line of text")
	parser.add_argument("--stdin", action="store_true", help="Read lines from stdin and synthesize each")
	args = parser.parse_args()

	if not args.text and not args.stdin:
		print("❌ Provide --text or --stdin")
		return 1

	async with AivisSpeechSynthesizer(endpoint=args.endpoint, speaker_id=args.speaker_id, volume=args.volume) as tts:
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


if __name__ == "__main__":
	asyncio.run(main())
