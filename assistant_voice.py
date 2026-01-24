"""
Local voice assistant entrypoint.

How to run:
  1) Install dependencies:
     pip install vosk sounddevice numpy
     # plus one Whisper backend:
     pip install faster-whisper   # recommended
     # or
     pip install -U openai-whisper
  2) Download a Vosk model and set MODEL_PATH below.
     Example: https://alphacephei.com/vosk/models
  3) Run:
     python assistant_voice.py
"""

from __future__ import annotations

import collections
import importlib
import json
import queue
import sys
import time
from dataclasses import dataclass
from typing import Deque

import numpy as np
import sounddevice as sd
import vosk

# ---------------------------
# Configuration parameters
# ---------------------------
MODEL_PATH = "models/vosk-model-small-fr-0.22"
WHISPER_MODEL_NAME = "small"  # used by faster-whisper or openai-whisper
SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
FRAME_MS = 20
PRE_ROLL_SEC = 1.0
MAX_CMD_SEC = 6.0
SILENCE_RMS_THRESHOLD = 350.0
SILENCE_DURATION_MS = 700
COOLDOWN_SEC = 1.5
HOTWORD = "jarvis"


@dataclass
class AudioChunk:
    data: bytes
    timestamp: float


def handle_command(text: str) -> None:
    """Stub to route commands to LLM/tools."""
    print(f"[command] {text}")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def strip_hotword(text: str, hotword: str) -> str:
    words = normalize_text(text).split()
    if not words:
        return ""
    if words[0] == hotword:
        return " ".join(words[1:]).strip()
    return normalize_text(text)


def compute_rms(audio_bytes: bytes) -> float:
    if not audio_bytes:
        return 0.0
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def build_whisper_model():
    """Load faster-whisper or openai-whisper, preferring faster-whisper."""
    if importlib.util.find_spec("faster_whisper") is not None:
        module = importlib.import_module("faster_whisper")
        return module.WhisperModel(WHISPER_MODEL_NAME, device="cpu")
    if importlib.util.find_spec("whisper") is not None:
        module = importlib.import_module("whisper")
        return module.load_model(WHISPER_MODEL_NAME)
    return None


def transcribe_with_whisper(model, audio_bytes: bytes) -> str:
    """Transcribe int16 mono 16k audio using the loaded Whisper backend."""
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if model.__class__.__module__.startswith("faster_whisper"):
        segments, _info = model.transcribe(samples, language="fr")
        text_parts = [segment.text for segment in segments]
        return " ".join(text_parts).strip()
    result = model.transcribe(samples, language="fr")
    return result.get("text", "").strip()


class RingBuffer:
    """Fixed-size byte ring buffer for pre-roll audio."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._chunks: Deque[bytes] = collections.deque()
        self._size_bytes = 0

    def append(self, data: bytes) -> None:
        if not data:
            return
        self._chunks.append(data)
        self._size_bytes += len(data)
        while self._size_bytes > self._max_bytes and self._chunks:
            removed = self._chunks.popleft()
            self._size_bytes -= len(removed)

    def get_bytes(self) -> bytes:
        return b"".join(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._size_bytes = 0


class VoiceAssistant:
    def __init__(self) -> None:
        self.queue: "queue.Queue[AudioChunk]" = queue.Queue()
        self.blocksize = int(SAMPLE_RATE * FRAME_MS / 1000)
        self.pre_roll_bytes = int(PRE_ROLL_SEC * SAMPLE_RATE * 2)
        self.ring_buffer = RingBuffer(self.pre_roll_bytes)
        self.model = vosk.Model(MODEL_PATH)
        self.recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        self.cooldown_until = 0.0
        self.whisper_model = build_whisper_model()

    def reset_recognizer(self) -> None:
        self.recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)

    def audio_callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            print(f"[audio] status: {status}", file=sys.stderr)
        self.queue.put(AudioChunk(bytes(indata), time.monotonic()))

    def detect_hotword(self, chunk: bytes) -> bool:
        # We rely on PartialResult for low latency: it updates every chunk without
        # waiting for an end-of-utterance, which is important for hotword detection.
        if self.recognizer.AcceptWaveform(chunk):
            result = json.loads(self.recognizer.Result())
            text = normalize_text(result.get("text", ""))
            return HOTWORD in text.split()
        partial = json.loads(self.recognizer.PartialResult())
        text = normalize_text(partial.get("partial", ""))
        return HOTWORD in text.split()

    def run(self) -> None:
        if self.whisper_model is None:
            print(
                "Whisper backend not available. Install faster-whisper or openai-whisper.",
                file=sys.stderr,
            )
            return

        capturing = False
        capture_buffer = bytearray()
        silence_ms = 0.0
        capture_start = 0.0

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self.blocksize,
            callback=self.audio_callback,
        ):
            print("Listening... say 'Jarvis' to start a command.")
            try:
                while True:
                    chunk = self.queue.get()
                    self.ring_buffer.append(chunk.data)

                    if not capturing:
                        if time.monotonic() < self.cooldown_until:
                            continue
                        if self.detect_hotword(chunk.data):
                            capturing = True
                            capture_start = time.monotonic()
                            silence_ms = 0.0
                            capture_buffer = bytearray()
                            capture_buffer.extend(self.ring_buffer.get_bytes())
                            capture_buffer.extend(chunk.data)
                    else:
                        capture_buffer.extend(chunk.data)
                        rms = compute_rms(chunk.data)
                        if rms < SILENCE_RMS_THRESHOLD:
                            silence_ms += FRAME_MS
                        else:
                            silence_ms = 0.0
                        elapsed = time.monotonic() - capture_start
                        if silence_ms >= SILENCE_DURATION_MS or elapsed >= MAX_CMD_SEC:
                            audio_bytes = bytes(capture_buffer)
                            transcript = transcribe_with_whisper(self.whisper_model, audio_bytes)
                            cleaned = strip_hotword(transcript, HOTWORD)
                            if cleaned:
                                handle_command(cleaned)
                            else:
                                print("[command] empty transcript")
                            capturing = False
                            self.cooldown_until = time.monotonic() + COOLDOWN_SEC
                            self.ring_buffer.clear()
                            self.reset_recognizer()
            except KeyboardInterrupt:
                print("Stopping.")
            except Exception as exc:  # noqa: BLE001 - top-level guard
                print(f"Error: {exc}", file=sys.stderr)


def main() -> None:
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
