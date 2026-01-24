#!/usr/bin/env python3
"""
Détection simple d'un mot-clé via Vosk (streaming micro).

Mot-clé détecté : "dis jarvis"
- Faible CPU
- Temps réel
- Idéal pour wake word
"""

import json
import queue
import sys

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# =====================
# CONFIG
# =====================
MODEL_PATH = "app/stt/models/vosk-model-small-fr-0.22"
SAMPLE_RATE = 16000
HOTWORD = "test"
BLOCK_SIZE = 4000
HOTWORD_GRAMMAR = [HOTWORD, f"dis {HOTWORD}"]
DETECTION_STREAK = 2

# =====================
# AUDIO CALLBACK
# =====================
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))


# =====================
# MAIN
# =====================
def main():
    print("Loading Vosk model...")
    model = Model(MODEL_PATH)

    recognizer = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(HOTWORD_GRAMMAR))
    recognizer.SetWords(False)

    print("Listening... (say 'Test')")

    detected_streak = 0
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = audio_queue.get()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
            else:
                result = json.loads(recognizer.PartialResult())
                text = result.get("partial", "").lower()

            if text:
                print(f"Heard: {text}")

            if HOTWORD in text:
                detected_streak += 1
            else:
                detected_streak = 0

            if detected_streak >= DETECTION_STREAK:
                print("\n🔥 HOTWORD DETECTED 🔥")
                print("Jarvis is listening...\n")
                break


if __name__ == "__main__":
    main()
